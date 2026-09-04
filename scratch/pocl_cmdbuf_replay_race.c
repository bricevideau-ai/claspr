/* pocl: minimal reproducer for the command-buffer replay race.
 *
 * Fixed by pocl commit "Command-buffers: retire the submission before
 * publishing CL_COMPLETE" (branch fix/cmdbuf-pending-race). Kept here as the
 * standalone artifact; pocl's own regression lives in
 * tests/runtime/test_command_buffer.c.
 *
 * Build/run:
 *   cc -O2 -I<pocl>/include -o cb_replay_race pocl_cmdbuf_replay_race.c -lOpenCL
 *   OCL_ICD_VENDORS=$HOME/.local/etc/OpenCL/vendors ./cb_replay_race 20000 200000
 *
 * argv[1] = replays, argv[2] = host-work spin between enqueue and wait.
 * Unpatched pocl: first failure at iteration ~6594 (spin 2000), ~897 (20000),
 * ~386 (200000) -- the failure rate scales with the window. spin 0 never fails.
 *
 * Original: cl_khr_command_buffer, sustained replay of ONE finalized
 * command-buffer fails CL_INVALID_OPERATION at a random iteration.
 *
 * The application waits on the completion event returned by
 * clEnqueueCommandBufferKHR before re-enqueueing, so the previous submission is
 * in CL_COMPLETE and this is NOT simultaneous use:
 *
 *   "Simultaneous Use:: When a command-buffer is submitted for execution
 *    without a prerequisite on all the previous submissions of the same
 *    command-buffer which are not in the CL_COMPLETE state."
 *   "... events ... would each be ways to avoid simultaneous use."
 *
 * so every iteration must succeed.
 *
 *   cc -o cb_replay_race cb_replay_race.c -lOpenCL && ./cb_replay_race [iters]
 */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { cl_int _e = (x); if (_e != CL_SUCCESS) { \
    fprintf(stderr, "%s:%d: %s -> %d\n", __FILE__, __LINE__, #x, _e); \
    return 2; } } while (0)

static const char *SRC =
    "__kernel void bump(__global int *d) { d[get_global_id(0)] += 1; }\n";

int main(int argc, char **argv)
{
    int iters = argc > 1 ? atoi(argv[1]) : 20000;
    /* Host work between enqueue and wait. THIS IS THE POINT: with no delay the
     * application is already blocked in clWaitForEvents when the buffer
     * completes, and pocl wakes it from notify_event_finished -- which runs
     * AFTER the command-buffer is retired, so the next enqueue is fine.
     *
     * A real application does work here (building result handles, bookkeeping).
     * If it arrives at clWaitForEvents AFTER the completion event has already
     * been set to CL_COMPLETE but BEFORE pocl retires the submission, it takes
     * the free event lock, sees CL_COMPLETE, never waits at all -- and its next
     * clEnqueueCommandBufferKHR is rejected. Tune with argv[2]. */
    long spin = argc > 2 ? atol(argv[2]) : 20000;

    cl_platform_id plat; cl_device_id dev;
    CHECK(clGetPlatformIDs(1, &plat, NULL));
    CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_DEFAULT, 1, &dev, NULL));

    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CHECK(err);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err); CHECK(err);

    cl_program prog = clCreateProgramWithSource(ctx, 1, &SRC, NULL, &err); CHECK(err);
    CHECK(clBuildProgram(prog, 1, &dev, NULL, NULL, NULL));
    cl_kernel k = clCreateKernel(prog, "bump", &err); CHECK(err);

    size_t n = 64;
    cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(cl_int), NULL, &err); CHECK(err);
    CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &buf));

#define EXT(name) \
    name##_fn p_##name = (name##_fn)clGetExtensionFunctionAddressForPlatform(plat, #name); \
    if (!p_##name) { fprintf(stderr, "no %s -- cl_khr_command_buffer missing\n", #name); return 77; }
    EXT(clCreateCommandBufferKHR)
    EXT(clCommandNDRangeKernelKHR)
    EXT(clFinalizeCommandBufferKHR)
    EXT(clEnqueueCommandBufferKHR)
    EXT(clReleaseCommandBufferKHR)
    EXT(clGetCommandBufferInfoKHR)

    /* No CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR: submissions are strictly serialised
     * by the blocking wait below, so it is not needed. */
    cl_command_buffer_khr cb = p_clCreateCommandBufferKHR(1, &q, NULL, &err); CHECK(err);
    /* >= 2 commands, as a real batched graph would have. */
    for (int i = 0; i < 3; ++i)
        CHECK(p_clCommandNDRangeKernelKHR(cb, NULL, NULL, k, 1, NULL, &n, NULL,
                                          0, NULL, NULL, NULL));
    CHECK(p_clFinalizeCommandBufferKHR(cb));

    /* Conformance check, independent of the race: the spec's three states are
     * Recording, Finalized and Executable. Finalized means "finalized but some
     * command still has unset arguments (mutable command-buffers only)". There
     * is no Pending state -- a fully specified finalized command-buffer is
     * Executable and stays Executable while a submission is in flight. */
    {
        cl_event ev;
        cl_command_buffer_state_khr st;
        CHECK(p_clEnqueueCommandBufferKHR(0, NULL, cb, 0, NULL, &ev));
        CHECK(p_clGetCommandBufferInfoKHR(cb, CL_COMMAND_BUFFER_STATE_KHR,
                                          sizeof(st), &st, NULL));
        printf("state while in flight: %u (%s; spec requires %u = Executable)\n",
               (unsigned)st,
               st == CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR ? "Executable"
               : st == CL_COMMAND_BUFFER_STATE_FINALIZED_KHR ? "Finalized -- WRONG"
               : "Recording -- WRONG",
               (unsigned)CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR);
        CHECK(clWaitForEvents(1, &ev));
        CHECK(clReleaseEvent(ev));
    }

    for (int i = 0; i < iters; ++i) {
        cl_event ev;
        cl_int e = p_clEnqueueCommandBufferKHR(0, NULL, cb, 0, NULL, &ev);
        if (e != CL_SUCCESS) {
            printf("FAILED at iteration %d: clEnqueueCommandBufferKHR -> %d%s\n",
                   i, e, e == CL_INVALID_OPERATION ? " (CL_INVALID_OPERATION)" : "");
            return 1;
        }
        volatile long sink = 0;
        for (long z = 0; z < spin; ++z) sink += z;
        (void)sink;
        CHECK(clWaitForEvents(1, &ev));   /* previous submission is now CL_COMPLETE */
        CHECK(clReleaseEvent(ev));
    }
    printf("OK: %d replays\n", iters);

    p_clReleaseCommandBufferKHR(cb);
    clReleaseMemObject(buf); clReleaseKernel(k); clReleaseProgram(prog);
    clReleaseCommandQueue(q); clReleaseContext(ctx);
    return 0;
}

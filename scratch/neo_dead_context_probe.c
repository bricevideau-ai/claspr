// Single-threaded probe of the hypothesis: completing a queued command's
// wait-list USER EVENT with a NEGATIVE status *terminates* that command, which
// per the OpenCL spec leaves the command-queue and its context in an
// implementation-defined / "no longer available" (dead) state.  Tests whether
// legacy Intel NEO actually kills the queue/context — which would explain why a
// blocking read gated on the terminated command never wakes (it's waiting on a
// dead queue), and why the defensive unmap returns CL_INVALID_VALUE.
//
// No threads, no race: enqueue an unmap gated on a user event, set the user
// event = -1 (terminating the unmap), then PROBE the queue/context with fresh,
// independent operations and report each status.
//
// Build: cc neo_dead_context_probe.c -lOpenCL -o probe
// Run:   OCL_ICD_VENDORS=/etc/OpenCL/vendors/intel_legacy1.icd ./probe
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
static const char *es(cl_int e) {
    switch (e) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
        return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    default: return "(other)";
    }
}
#define MUST(expr)                                                            \
    do {                                                                      \
        cl_int _e = (expr);                                                   \
        if (_e != CL_SUCCESS) {                                               \
            fprintf(stderr, "SETUP ERR %s at %d (%s)\n", es(_e), __LINE__,    \
                    #expr);                                                   \
            exit(2);                                                          \
        }                                                                     \
    } while (0)

int main(void) {
    cl_platform_id plat;
    cl_device_id dev;
    MUST(clGetPlatformIDs(1, &plat, NULL));
    MUST(clGetDeviceIDs(plat, CL_DEVICE_TYPE_DEFAULT, 1, &dev, NULL));
    char name[256] = {0}, drv[256] = {0};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
    clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof(drv), drv, NULL);
    fprintf(stderr, "Device: %s | driver %s\n\n", name, drv);

    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    MUST(err);
    cl_command_queue q = clCreateCommandQueue(
        ctx, dev, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    MUST(err);
    cl_mem buf =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * sizeof(int), NULL, &err);
    MUST(err);
    int zero = 0;
    MUST(clEnqueueFillBuffer(q, buf, &zero, sizeof(zero), 0, N * sizeof(int), 0,
                             NULL, NULL));
    MUST(clFinish(q));

    void *ptr = clEnqueueMapBuffer(q, buf, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                                   0, N * sizeof(int), 0, NULL, NULL, &err);
    MUST(err);

    cl_event U = clCreateUserEvent(ctx, &err);
    MUST(err);
    cl_event unmap_ev;
    MUST(clEnqueueUnmapMemObject(q, buf, ptr, 1, &U, &unmap_ev));
    fprintf(stderr, "enqueued unmap gated on user event U\n");

    // Terminate the queued unmap by completing its dependency negative.
    fprintf(stderr, "clSetUserEventStatus(U, -1) -> terminate the unmap\n");
    MUST(clSetUserEventStatus(U, -1));

    // What status did the terminated unmap take?
    cl_int unmap_status = 0;
    clGetEventInfo(unmap_ev, CL_EVENT_COMMAND_EXECUTION_STATUS,
                   sizeof(unmap_status), &unmap_status, NULL);
    fprintf(stderr, "unmap event exec status = %d (%s)\n\n", unmap_status,
            unmap_status < 0 ? es(unmap_status) : "non-negative");

    // ---- PROBE the queue/context for liveness with fresh, independent ops ----
    fprintf(stderr, "=== probing queue/context liveness ===\n");

    // (a) A fresh independent buffer + fill on the SAME queue.
    cl_mem buf2 =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * sizeof(int), NULL, &err);
    fprintf(stderr, "create fresh buffer: %s\n", es(err));
    cl_int fe = clEnqueueFillBuffer(q, buf2, &zero, sizeof(zero), 0,
                                    N * sizeof(int), 0, NULL, NULL);
    fprintf(stderr, "fill fresh buffer on same queue: %s\n", es(fe));

    // (b) clFinish the queue — does the queue still drain?
    cl_int fin = clFinish(q);
    fprintf(stderr, "clFinish(queue): %s\n", es(fin));

    // (c) A fresh non-blocking read of buf2 + wait.
    int *host = malloc(N * sizeof(int));
    cl_int re = clEnqueueReadBuffer(q, buf2, CL_TRUE, 0, N * sizeof(int), host,
                                    0, NULL, NULL);
    fprintf(stderr, "blocking read of fresh buffer: %s\n", es(re));

    fprintf(stderr, "\nVERDICT: %s\n",
            (fe == CL_SUCCESS && fin == CL_SUCCESS && re == CL_SUCCESS)
                ? "queue/context still ALIVE after terminating a queued command"
                : "queue/context DEAD/degraded after terminating a queued command");
    free(host);
    return 0;
}

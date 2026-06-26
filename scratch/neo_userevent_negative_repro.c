// Repro: a BLOCKING clEnqueueReadBuffer whose wait-list transitively depends on
// a USER EVENT that is completed with a NEGATIVE status.  Does the blocking call
// return, or hang?  Mirrors the claspr `and_then_host` error-path shape:
//
//   map (non-blocking)  -- event M
//   user_event U (host-controlled)
//   unmap   gated on U           -- event UN
//   BLOCKING read  gated on UN   <-- the call under test
//   [worker thread] eventually: clSetUserEventStatus(U, -1)
//
// Two findings this reproduces:
//  (1) THE HANG: if the blocking read is already parked in its wait when U is
//      set NEGATIVE, legacy Intel NEO never wakes it (driver 24.35.30872.36).
//      pocl returns CL_SUCCESS; rusticl returns -14 (cascade) — both RETURN.
//  (2) THE DOUBLE-UNMAP: the claspr error path also issues a second, un-gated
//      defensive unmap on the worker thread -> CL_INVALID_VALUE (buffer already
//      unmapped).  Reproduced here in --double-unmap mode.
//
// The outcome is SCHEDULE-DEPENDENT (intermittent in the wild).  This repro
// FORCES the hang schedule deterministically: the worker waits on the map event
// AND a fixed delay, so the main thread's blocking read is committed to its wait
// before U goes negative.  Pass --race to instead use a tiny delay (the
// usually-passing schedule) to show the contrast.
//
// Build:  cc neo_userevent_negative_repro.c -lOpenCL -lpthread -o repro
// Run:    OCL_ICD_VENDORS=/etc/OpenCL/vendors/intel_legacy1.icd ./repro
//         (add --race to show the passing schedule; --double-unmap for finding 2)
// Exit:   0 = read returned (no hang) | 42 = HANG (watchdog) | 2 = setup error
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define N 1024
#define WATCHDOG_SECS 8

static cl_event g_map_ev;       // worker waits this (map complete) before acting
static cl_event g_user_event;   // the user event the unmap gates on
static cl_command_queue g_queue;
static cl_mem g_buf;
static void *g_ptr;
static cl_context g_ctx;
static cl_device_id g_dev;
static int g_force_hang = 1;    // 1 = force hang schedule, 0 = --race
static int g_double_unmap = 0;  // 1 = also do the defensive un-gated unmap
static int g_nonblock = 0;      // 1 = non-blocking read + clWaitForEvents
static int g_fixed = 0;         // 1 = proposed fix: cancel, drop queue, unmap fresh

#define CHECK(expr)                                                            \
    do {                                                                       \
        cl_int _e = (expr);                                                    \
        if (_e != CL_SUCCESS) {                                                \
            fprintf(stderr, "ERR %d at %s:%d (%s)\n", _e, __FILE__, __LINE__,  \
                    #expr);                                                    \
            exit(2);                                                           \
        }                                                                      \
    } while (0)

static void *worker(void *arg) {
    (void)arg;
    // Mirror the real worker: wait the map event (closure would run here).
    clWaitForEvents(1, &g_map_ev);

    if (g_fixed) {
        // PROPOSED FIX ORDER: cancel FIRST (terminate the gated unmap), drop the
        // queue, THEN unmap on a fresh queue — so two unmaps are never
        // registered on the buffer at the same time.
        fprintf(stderr, "[worker] (fixed) clSetUserEventStatus(U,-1) FIRST\n");
        clSetUserEventStatus(g_user_event, -1); // cancels the gated unmap
        // DRAIN the queue before dropping it — let the terminated unmap (and any
        // dependents) settle so we don't release a queue with in-flight work.
        cl_int finr = clFinish(g_queue);
        fprintf(stderr, "[worker] (fixed) clFinish(old queue) -> %d\n", finr);
        fprintf(stderr, "[worker] (fixed) drop queue, make a fresh one\n");
        clReleaseCommandQueue(g_queue);
        cl_int qe;
        cl_command_queue q2 = clCreateCommandQueue(g_ctx, g_dev, 0, &qe);
        // Now unmap on the fresh queue — nothing else references the buffer.
        cl_int de = clEnqueueUnmapMemObject(q2, g_buf, g_ptr, 0, NULL, NULL);
        fprintf(stderr, "[worker] (fixed) unmap on fresh queue -> %d (%s)\n", de,
                de == CL_SUCCESS ? "CL_SUCCESS" : "non-success");
        clFinish(q2);
        return NULL;
    }

    // CURRENT (buggy) ORDER: defensive un-gated unmap is registered while the
    // gated unmap is STILL LIVE (U not yet negative) -> two unmaps on the same
    // buffer at once; NEO already took map_count -> 0 at the gated unmap's
    // ENQUEUE, so this returns CL_INVALID_VALUE. THEN we cancel.
    if (g_double_unmap) {
        cl_int de = clEnqueueUnmapMemObject(g_queue, g_buf, g_ptr, 0, NULL, NULL);
        fprintf(stderr, "[worker] defensive un-gated unmap -> %d%s\n", de,
                de == CL_INVALID_VALUE ? " (CL_INVALID_VALUE: double unmap)" : "");
    }
    if (g_force_hang) {
        // Guarantee the main thread's blocking read is already waiting.
        usleep(200000); // 200ms
    } else {
        usleep(500);    // tiny: usually lets the read win the race (passing sched)
    }
    fprintf(stderr, "[worker] clSetUserEventStatus(U, -1)\n");
    clSetUserEventStatus(g_user_event, -1);
    fprintf(stderr, "[worker] U set negative\n");
    return NULL;
}

static void *watchdog(void *arg) {
    (void)arg;
    sleep(WATCHDOG_SECS);
    fprintf(stderr,
            "[watchdog] %ds with no return from blocking read -> HANG REPRODUCED\n",
            WATCHDOG_SECS);
    _exit(42);
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--race")) g_force_hang = 0;
        else if (!strcmp(argv[i], "--double-unmap")) g_double_unmap = 1;
        else if (!strcmp(argv[i], "--nonblock")) g_nonblock = 1;
        else if (!strcmp(argv[i], "--fixed")) { g_fixed = 1; g_double_unmap = 1; }
    }

    cl_platform_id plat;
    cl_device_id dev;
    CHECK(clGetPlatformIDs(1, &plat, NULL));
    CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_DEFAULT, 1, &dev, NULL));

    char name[256] = {0}, ver[256] = {0}, drv[256] = {0};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof(ver), ver, NULL);
    clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof(drv), drv, NULL);
    fprintf(stderr, "Device: %s | %s | driver %s\n", name, ver, drv);
    fprintf(stderr, "Mode: %s%s\n", g_force_hang ? "force-hang" : "race",
            g_double_unmap ? " +double-unmap" : "");

    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    CHECK(err);
    g_ctx = ctx;
    g_dev = dev;
    // OUT-OF-ORDER queue — this is what claspr's terminal uses, and is the
    // missing ingredient: in an in-order queue commands serialize by submission
    // order so the event gating is moot; only OOO genuinely blocks the read on
    // the (negatively-resolved) event.
    g_queue = clCreateCommandQueue(ctx, dev,
                                   CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    CHECK(err);
    g_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * sizeof(int), NULL, &err);
    CHECK(err);
    int zero = 0;
    CHECK(clEnqueueFillBuffer(g_queue, g_buf, &zero, sizeof(zero), 0,
                              N * sizeof(int), 0, NULL, NULL));
    CHECK(clFinish(g_queue));

    // 1. Non-blocking map -> g_map_ev.
    // Match the real trace: map READ|WRITE (claspr maps mutable views RW).
    g_ptr = clEnqueueMapBuffer(g_queue, g_buf, CL_FALSE,
                               CL_MAP_READ | CL_MAP_WRITE, 0, N * sizeof(int), 0,
                               NULL, &g_map_ev, &err);
    CHECK(err);
    // 2. User event the unmap gates on.
    g_user_event = clCreateUserEvent(ctx, &err);
    CHECK(err);
    // 3. Unmap gated on the user event -> unmap_ev.
    cl_event unmap_ev;
    CHECK(clEnqueueUnmapMemObject(g_queue, g_buf, g_ptr, 1, &g_user_event,
                                  &unmap_ev));

    pthread_t wt, wd;
    pthread_create(&wt, NULL, worker, NULL);
    pthread_create(&wd, NULL, watchdog, NULL);

    // 4. BLOCKING read gated on the unmap (transitively on the negative U).
    int *host = malloc(N * sizeof(int));
    cl_int read_err;
    if (g_nonblock) {
        // Non-blocking read + explicit clWaitForEvents on its event — a
        // different driver path than a blocking enqueue.
        cl_event read_ev;
        fprintf(stderr, "[main] non-blocking clEnqueueReadBuffer...\n");
        read_err = clEnqueueReadBuffer(g_queue, g_buf, CL_FALSE, 0,
                                       N * sizeof(int), host, 1, &unmap_ev,
                                       &read_ev);
        fprintf(stderr, "[main] enqueue returned %d; clWaitForEvents...\n",
                read_err);
        cl_int we = clWaitForEvents(1, &read_ev);
        fprintf(stderr, "[main] clWaitForEvents RETURNED, status %d\n", we);
        read_err = we;
    } else {
        fprintf(stderr, "[main] entering BLOCKING clEnqueueReadBuffer...\n");
        read_err = clEnqueueReadBuffer(g_queue, g_buf, CL_TRUE, 0,
                                       N * sizeof(int), host, 1, &unmap_ev, NULL);
        fprintf(stderr, "[main] blocking read RETURNED, status %d\n", read_err);
    }
    printf("NO HANG: read returned status %d\n", read_err);

    pthread_join(wt, NULL);
    free(host);
    return 0;
}

// Reproducer for the thread-safety bug in pocl's clReleaseEvent: concurrent,
// individually-legal release of the same MARKER/command event by an application
// thread and by pocl's async-callback worker thread.
//
// ROOT CAUSE (pocl, release build, USE_POCL_MEMMANAGER=OFF):
//   clReleaseEvent() checks validity UNLOCKED -> in a release build that is just
//   `event != NULL` (the magic-marker check is debug-only). It then takes the
//   per-event mutex `event->pocl_lock`, which lives INSIDE the event struct. The
//   refcount==0 path UNLOCKS, POCL_DESTROY_OBJECT (pthread_mutex_destroy on the
//   embedded mutex) and pocl_mem_manager_free_event (a plain free() when the
//   mem-manager is off). Two legitimate concurrent releases of the same event
//   -- one driving refcount->0 + free, the other between its unlocked NULL check
//   and POCL_LOCK_OBJ -- make the second thread lock a destroyed/freed mutex:
//     PTHREAD ERROR in POclReleaseEvent():36: Invalid argument (22)  (SIGABRT)
//   or a use-after-free (SIGSEGV).
//
// This mirrors what claspr hits ~23% on its async .run().await error path: a
// CL_COMPLETE callback is registered on a marker event; when the marker
// completes, pocl retains the event and hands it to its async-callback worker
// thread (pocl_event_cb_push), which fires the callback then clReleaseEvent()s
// its reference (process_event_cb in lib/CL/pocl_util.c). Concurrently the
// application releases its own reference. Both releases are individually correct
// and the refcount arithmetic is balanced; the bug is the concurrent
// free-vs-enter race in clReleaseEvent.
//
// Shape here (closest to the real path, no claspr/Rust): per iteration enqueue a
// marker, retain it once for the application, register a CL_COMPLETE callback,
// then race -- the application thread releases its reference in a tight loop
// while pocl's worker thread completes the marker and releases the callback
// reference. No barrier/sleep: the application keeps re-arming so that, across
// iterations, its release lands inside the worker's clReleaseEvent free window.
//
// Build: cc -O2 pocl_marker_release_race.c -lOpenCL -lpthread -o repro
// Run:   OCL_ICD_VENDORS=<pocl vendors dir> ./repro
// Exit:  0 = survived (no crash this run; re-run / raise ITERS)
//        134 (SIGABRT) / 139 (SIGSEGV) = bug reproduced.
//
// NOTE: this pure-C repro is timing-sensitive and did NOT reliably crash clean
// pocl in our runs (the window is a handful of instructions; the worker-thread
// scheduling rarely overlaps it from C). The reliable reproducer for this bug
// is the claspr Rust test `await_propagates_chain_error` (~23-27% SIGSEGV on
// unfixed pocl). This file is kept as the documented shape for an upstream bug
// report.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ITERS 2000000

#define CHECK(expr)                                                            \
  do {                                                                         \
    cl_int _e = (expr);                                                        \
    if (_e != CL_SUCCESS) {                                                    \
      fprintf (stderr, "setup err %d at line %d\n", _e, __LINE__);             \
      exit (2);                                                                \
    }                                                                          \
  } while (0)

static void CL_CALLBACK on_complete (cl_event ev, cl_int status, void *ud) {
  (void)ev; (void)status; (void)ud; /* non-blocking: just lets the worker go */
}

int main (void) {
  cl_platform_id plat;
  cl_device_id dev;
  CHECK (clGetPlatformIDs (1, &plat, NULL));
  CHECK (clGetDeviceIDs (plat, CL_DEVICE_TYPE_DEFAULT, 1, &dev, NULL));
  char name[256] = {0}, drv[256] = {0};
  clGetDeviceInfo (dev, CL_DEVICE_NAME, sizeof (name), name, NULL);
  clGetDeviceInfo (dev, CL_DRIVER_VERSION, sizeof (drv), drv, NULL);
  fprintf (stderr, "Device: %s | driver %s\n", name, drv);

  cl_int err;
  cl_context ctx = clCreateContext (NULL, 1, &dev, NULL, NULL, &err);
  CHECK (err);
  cl_command_queue q = clCreateCommandQueue (ctx, dev, 0, &err);
  CHECK (err);

  for (long it = 0; it < ITERS; it++) {
    cl_event ev;
    CHECK (clEnqueueMarkerWithWaitList (q, 0, NULL, &ev));
    CHECK (clRetainEvent (ev));                       /* application reference */
    CHECK (clSetEventCallback (ev, CL_COMPLETE, on_complete, NULL));
    CHECK (clFlush (q));      /* worker will complete it + release its ref     */
    /* Release the enqueue reference and the application reference back-to-back,
     * with no sync: somewhere across iterations one of these lands in the
     * worker thread's clReleaseEvent free window. */
    clReleaseEvent (ev);
    clReleaseEvent (ev);

    if ((it & 0x3ffff) == 0)
      fprintf (stderr, "iter %ld\n", it);
  }

  clReleaseCommandQueue (q);
  clReleaseContext (ctx);
  fprintf (stderr, "survived %ld iterations (no crash this run)\n", (long)ITERS);
  return 0;
}

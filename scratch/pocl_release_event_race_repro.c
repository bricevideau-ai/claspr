// Reproducer: pocl's clReleaseEvent is not safe against two concurrent
// releases of the same event that bring its refcount to zero. The per-event
// mutex (pocl_lock) lives INSIDE the event struct; clReleaseEvent checks
// validity UNLOCKED (in a release build that is just `!= NULL`), then takes
// that lock. When two threads release the same event concurrently — one drives
// the refcount to 0 and frees + destroys the embedded mutex + recycles the
// memory, while the other is between its validity check and POCL_LOCK_OBJ — the
// second locks a destroyed/recycled mutex:
//   PTHREAD ERROR in POclReleaseEvent():36: Invalid argument (22)   (SIGABRT)
// or a use-after-free (SIGSEGV).
//
// This mirrors a legal OpenCL usage that claspr hits on its async
// `.run().await` error path: a CL_COMPLETE callback is registered on a marker
// event, the marker completes on pocl's worker thread (which releases pocl's
// own references there), concurrently with the application releasing its own
// reference from another thread. Both releases are legitimate and individually
// correct — the refcount arithmetic is balanced (POCL_DEBUG=refcounts shows no
// leak/imbalance) — but pocl's clReleaseEvent races itself at the free.
//
// To force the race deterministically without the rest of a runtime, this repro
// has N worker threads each: create an event (a user event stands in for any
// event; the bug is in the shared clReleaseEvent free path), retain it to
// refcount 2, then have TWO threads clReleaseEvent the SAME event at the same
// time (synchronised on a barrier) so the two releases reach refcount 0
// concurrently. Repeated in a tight loop to hit the window.
//
// Build: cc pocl_release_event_race_repro.c -lOpenCL -lpthread -o repro
// Run:   OCL_ICD_VENDORS=<pocl vendors dir> ./repro
// Exit:  0 = survived all iterations (no race observed this run; re-run / raise
//        ITERS) ; SIGABRT/SIGSEGV = bug reproduced.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define PAIRS 64       /* concurrent release-pairs per iteration */
#define ITERS 20000    /* loop to hit the timing window */

static cl_context g_ctx;
static cl_event g_events[PAIRS];
static pthread_barrier_t g_start; /* releases both racers simultaneously */

#define CHECK(expr)                                                            \
  do {                                                                         \
    cl_int _e = (expr);                                                        \
    if (_e != CL_SUCCESS) {                                                    \
      fprintf(stderr, "setup err %d at %d\n", _e, __LINE__);                   \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

// Releaser thread: wait the barrier, then release every event once. Two such
// threads run, so each event (refcount 2) gets two concurrent releases → 0.
static void *releaser(void *arg) {
  (void)arg;
  pthread_barrier_wait(&g_start);
  for (int i = 0; i < PAIRS; i++)
    clReleaseEvent(g_events[i]); /* may race the other thread to refcount 0 */
  return NULL;
}

int main(void) {
  cl_platform_id plat;
  cl_device_id dev;
  CHECK(clGetPlatformIDs(1, &plat, NULL));
  CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_DEFAULT, 1, &dev, NULL));
  char name[256] = {0}, drv[256] = {0};
  clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
  clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof(drv), drv, NULL);
  fprintf(stderr, "Device: %s | driver %s\n", name, drv);

  cl_int err;
  g_ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
  CHECK(err);

  for (int it = 0; it < ITERS; it++) {
    // Build PAIRS events, each at refcount 2 (created =1, retained =2), so two
    // releaser threads each releasing once drive each event to 0 concurrently.
    for (int i = 0; i < PAIRS; i++) {
      g_events[i] = clCreateUserEvent(g_ctx, &err);
      CHECK(err);
      CHECK(clRetainEvent(g_events[i])); /* refcount 1 -> 2 */
    }
    pthread_barrier_init(&g_start, NULL, 2);
    pthread_t a, b;
    pthread_create(&a, NULL, releaser, NULL);
    pthread_create(&b, NULL, releaser, NULL);
    pthread_join(a, NULL);
    pthread_join(b, NULL);
    pthread_barrier_destroy(&g_start);
    if ((it & 0x3ff) == 0)
      fprintf(stderr, "iter %d ok\n", it);
  }

  fprintf(stderr, "survived %d iterations\n", ITERS);
  printf("NO RACE OBSERVED (re-run or raise ITERS)\n");
  clReleaseContext(g_ctx);
  return 0;
}

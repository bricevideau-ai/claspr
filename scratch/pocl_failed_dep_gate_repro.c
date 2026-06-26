// Reproducer: a command whose wait-list event is completed with a NEGATIVE
// (failed) status is executed anyway instead of being terminated, leading to a
// crash (SIGSEGV in a pocl worker thread: pocl_exec_command -> memcpy) on
// memory the terminated dependency already released.
//
// Spec (OpenCL 3.0 §5.10, clSetUserEventStatus / event execution status):
// completing an event with a negative value terminates it; commands in the same
// or other queues that wait on it (and the event itself) are terminated, with
// the error propagated. Such a waiting command must NOT execute.
//
// ---------------------------------------------------------------------------
// Why this exact shape (and what it deliberately avoids)
// ---------------------------------------------------------------------------
// This mirrors a real host-callback pattern: a host thread runs a callback on a
// mapped buffer and, on failure, must abort the rest of an already-enqueued
// graph. The pieces:
//
//   start   : user event gating the FIRST enqueue. Released only AFTER the whole
//             graph is enqueued, so the graph is fully committed before anything
//             runs — this makes the bug DETERMINISTIC rather than a timing race.
//   fire    : user event gating the unmap; always completed CL_COMPLETE (one
//             clean unmap — no second/defensive unmap).
//   proceed : user event gating the final read; completed NEGATIVE on the error
//             path to abort the downstream read (CL_COMPLETE on the --success
//             control path).
//
//   Graph (one out-of-order queue):
//     fill  (gated on [start])
//     map   (gated on [fill])              -> mapev (non-blocking map)
//     unmap (gated on [fire])              -> un
//     read  (gated on [un, proceed])       -> read_ev   (LAST command)
//   worker thread: wait mapev; fire=CL_COMPLETE; proceed = (-1 | CL_COMPLETE)
//   main thread:   release start=CL_COMPLETE; clWaitForEvents(read_ev)
//
// It uses clWaitForEvents on the last event (NOT clFinish) on purpose, to avoid
// conflating this bug with a separate clFinish-on-a-terminated-command issue.
// There is one CLI control only — `--success` — which completes `proceed`
// normally; that path must always work and is the baseline.
//
// ---------------------------------------------------------------------------
// Expected vs observed
// ---------------------------------------------------------------------------
//   Correct:  the read is terminated (its execution status is negative,
//             CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST); it does NOT run.
//             Program prints PASS and exits 0.
//   Bug:      the read executes on the buffer the terminated chain released ->
//             SIGSEGV in a pocl worker thread; or it "succeeds" (status
//             CL_COMPLETE) having run on terminated data -> prints FAIL.
//
// A watchdog thread aborts after a few seconds (exit 42) so the program can
// never hang the harness; this bug manifests as a crash, not a hang, on pocl.
//
// Build: cc pocl_failed_dep_gate_repro.c -lOpenCL -lpthread -o repro
// Run:   OCL_ICD_VENDORS=<pocl OpenCL/vendors dir> ./repro
//        OCL_ICD_VENDORS=<...>                      ./repro --success   (control)
// Exit:  0 = correct (read terminated / control completed); 1 = FAIL (read ran
//        on terminated data); 42 = watchdog (hang); 2 = setup error; SIGSEGV = bug.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define N 1024
#define WATCHDOG_SECS 8

static cl_event fire, proceed, mapev;
static cl_command_queue Q;
static int err_path = 1; /* default: abort path (proceed = -1) */

#define CHECK(expr)                                                            \
  do {                                                                         \
    cl_int _e = (expr);                                                        \
    if (_e != CL_SUCCESS) {                                                    \
      fprintf(stderr, "setup error %d at line %d (%s)\n", _e, __LINE__, #expr);\
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

/* Host worker: waits for the map to complete, fires the unmap, then either
   aborts (proceed = -1) or proceeds (proceed = CL_COMPLETE). */
static void *worker(void *arg) {
  (void)arg;
  clWaitForEvents(1, &mapev); /* map completes only after `start` is released */
  clSetUserEventStatus(fire, CL_COMPLETE);
  clSetUserEventStatus(proceed, err_path ? -1 : CL_COMPLETE);
  return NULL;
}

static void *watchdog(void *arg) {
  (void)arg;
  sleep(WATCHDOG_SECS);
  fprintf(stderr, "[watchdog] %ds elapsed with no return -> HANG\n",
          WATCHDOG_SECS);
  _exit(42);
}

int main(int argc, char **argv) {
  for (int i = 1; i < argc; i++)
    if (!strcmp(argv[i], "--success"))
      err_path = 0;

  cl_platform_id plat;
  cl_device_id dev;
  CHECK(clGetPlatformIDs(1, &plat, NULL));
  CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_DEFAULT, 1, &dev, NULL));
  char name[256] = {0}, drv[256] = {0};
  clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
  clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof(drv), drv, NULL);
  fprintf(stderr, "Device: %s | driver %s | path: %s\n", name, drv,
          err_path ? "abort (proceed=-1)" : "success (proceed=CL_COMPLETE)");

  cl_int err;
  cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
  CHECK(err);
  /* Out-of-order queue: commands wait on their event dependencies rather than
     submission order, so the gating is meaningful. */
  Q = clCreateCommandQueue(ctx, dev, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                           &err);
  CHECK(err);
  cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * sizeof(int), NULL,
                              &err);
  CHECK(err);

  /* `start` gates the first enqueue so the whole graph is committed before any
     of it runs. */
  cl_event start = clCreateUserEvent(ctx, &err);
  CHECK(err);

  int seven = 7;
  cl_event fillev;
  CHECK(clEnqueueFillBuffer(Q, buf, &seven, sizeof(seven), 0, N * sizeof(int), 1,
                            &start, &fillev));
  void *ptr = clEnqueueMapBuffer(Q, buf, CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0,
                                 N * sizeof(int), 1, &fillev, &mapev, &err);
  CHECK(err);

  fire = clCreateUserEvent(ctx, &err);
  CHECK(err);
  proceed = clCreateUserEvent(ctx, &err);
  CHECK(err);
  cl_event un;
  CHECK(clEnqueueUnmapMemObject(Q, buf, ptr, 1, &fire, &un));

  pthread_t wt, wd;
  pthread_create(&wt, NULL, worker, NULL);
  pthread_create(&wd, NULL, watchdog, NULL);

  /* LAST command: a read gated on [unmap, proceed]. On the abort path `proceed`
     becomes negative, so this read must be terminated, not executed. */
  cl_event wait_list[2] = {un, proceed};
  int *host = malloc(N * sizeof(int));
  cl_event read_ev;
  cl_int enq = clEnqueueReadBuffer(Q, buf, CL_FALSE, 0, N * sizeof(int), host, 2,
                                   wait_list, &read_ev);
  fprintf(stderr, "[main] whole graph enqueued (read enqueue = %d). "
                  "Releasing start.\n",
          enq);

  /* Whole graph is enqueued -> release it. */
  clSetUserEventStatus(start, CL_COMPLETE);

  /* Wait on the LAST event only (deliberately not clFinish). */
  fprintf(stderr, "[main] clWaitForEvents(read_ev)...\n");
  cl_int we = clWaitForEvents(1, &read_ev);
  fprintf(stderr, "[main] clWaitForEvents returned %d\n", we);

  /* Join the worker before teardown so its CL calls finish before the context
     is destroyed. */
  pthread_join(wt, NULL);

  cl_int read_status = 0;
  clGetEventInfo(read_ev, CL_EVENT_COMMAND_EXECUTION_STATUS,
                 sizeof(read_status), &read_status, NULL);
  fprintf(stderr, "[main] read command execution status = %d\n", read_status);

  int rc;
  if (!err_path) {
    /* control: success path must complete normally */
    rc = (read_status == CL_COMPLETE) ? 0 : 1;
    fprintf(stderr, rc == 0 ? "PASS (control): read completed.\n"
                            : "FAIL (control): read did not complete.\n");
  } else if (read_status < 0) {
    fprintf(stderr, "PASS: read terminated (not executed) — correct.\n");
    rc = 0;
  } else {
    fprintf(stderr, "FAIL: read ran despite a failed dependency "
                    "(status=%d) — executed on terminated data.\n",
            read_status);
    rc = 1;
  }
  printf("%s\n", rc == 0 ? "PASS" : "FAIL");
  return rc;
}

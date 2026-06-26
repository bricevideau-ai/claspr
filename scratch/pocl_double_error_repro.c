/* Reproducer: a plain command (clEnqueueReadBuffer) whose wait list contains
 * TWO user events that are both set to an ERROR status concurrently from two
 * threads. Neither dependency completes successfully and the waiter is not a
 * marker/barrier -- yet the two failed-dependency broadcasts both call
 * pocl_update_event_finished on the same waiter, double-finishing it.
 *
 * This isolates the general case: the double-finish is NOT specific to
 * marker/barrier and NOT dependent on a completion-vs-error race. Two errors
 * on any multi-dependency command suffice.
 *
 * Build:  gcc -O2 -o pocl_double_error_repro pocl_double_error_repro.c -lOpenCL -lpthread
 * Expect (buggy pocl):  intermittent  pocl_util.c: Assertion 'event->status > CL_COMPLETE' failed
 * Expect (fixed pocl):  "PASS: read terminated, no double-finish"
 */
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define N 1024

static cl_event eA, eB;       /* the two user-event dependencies            */
static cl_event start;        /* gates the read enqueue (commit before fire) */

/* Two worker threads, one per user event, both firing an ERROR at the same
 * time so their broadcasts into the shared waiter race. */
static void *fire_A (void *p) { (void)p; clWaitForEvents (1, &start);
                                clSetUserEventStatus (eA, -1); return NULL; }
static void *fire_B (void *p) { (void)p; clWaitForEvents (1, &start);
                                clSetUserEventStatus (eB, -1); return NULL; }
static void *watchdog (void *p){ (void)p; sleep (10);
                                 fprintf (stderr, "HANG\n"); _exit (42); }

int main (void)
{
  cl_platform_id plat; cl_device_id dev; cl_int err;
  clGetPlatformIDs (1, &plat, NULL);
  clGetDeviceIDs (plat, CL_DEVICE_TYPE_DEFAULT, 1, &dev, NULL);
  cl_context ctx = clCreateContext (NULL, 1, &dev, NULL, NULL, &err);
  cl_command_queue Q =
    clCreateCommandQueue (ctx, dev, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
  cl_mem buf = clCreateBuffer (ctx, CL_MEM_READ_WRITE, N * 4, NULL, &err);

  start = clCreateUserEvent (ctx, &err);
  eA    = clCreateUserEvent (ctx, &err);
  eB    = clCreateUserEvent (ctx, &err);

  /* A no-op command gated on `start` so the graph is committed before we fire,
   * mirroring the start-gate timing that widens the race window. */
  cl_event gate_ev;
  clEnqueueMarkerWithWaitList (Q, 1, &start, &gate_ev);

  /* The waiter: a PLAIN read-buffer depending on BOTH user events. */
  int *host = malloc (N * 4);
  cl_event wl[2] = { eA, eB };
  cl_event read_ev;
  cl_int enq = clEnqueueReadBuffer (Q, buf, CL_FALSE, 0, N * 4, host, 2, wl, &read_ev);
  fprintf (stderr, "[main] read enqueue = %d; releasing start, both deps -> error\n", enq);

  pthread_t tA, tB, tw;
  pthread_create (&tw, NULL, watchdog, NULL);
  pthread_create (&tA, NULL, fire_A, NULL);
  pthread_create (&tB, NULL, fire_B, NULL);

  clSetUserEventStatus (start, CL_COMPLETE);

  cl_int w = clWaitForEvents (1, &read_ev);
  fprintf (stderr, "[main] clWaitForEvents(read_ev) = %d\n", w);

  pthread_join (tA, NULL);
  pthread_join (tB, NULL);

  cl_int st = 0;
  clGetEventInfo (read_ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof (st), &st, NULL);
  fprintf (stderr, "[main] read exec status = %d\n", st);

  if (st < 0)
    { printf ("PASS: read terminated, no double-finish\n"); return 0; }
  printf ("UNEXPECTED: read status %d\n", st);
  return 1;
}

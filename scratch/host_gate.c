// Option 2: NO negative user event ever. `fire` always CL_COMPLETE (unmap runs).
// Abort is decided HOST-SIDE: the main thread waits for the unmap to finish,
// checks an error flag, and only enqueues the downstream read if no error.
// On error: read is simply never enqueued. No user event is ever set negative,
// so there is no lost-wakeup race for the driver to hit.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdatomic.h>
#define N 1024
static cl_event fire, mapev, unmap_ev; static cl_command_queue Q;
static atomic_int err_flag=0; static int err_path=1;
static void* wk(void*a){(void)a; clWaitForEvents(1,&mapev);
  // closure "fails": set flag, then ALWAYS fire unmap CL_COMPLETE (no negative).
  if(err_path) atomic_store(&err_flag,1);
  clSetUserEventStatus(fire, CL_COMPLETE);
  return NULL;}
static void* wd(void*a){(void)a; sleep(8); fprintf(stderr,"HANG\n"); _exit(42);}
int main(int argc,char**argv){
  for(int i=1;i<argc;i++) if(!strcmp(argv[i],"--success")) err_path=0;
  cl_platform_id p; cl_device_id d; clGetPlatformIDs(1,&p,NULL); clGetDeviceIDs(p,CL_DEVICE_TYPE_DEFAULT,1,&d,NULL);
  cl_int e; cl_context c=clCreateContext(NULL,1,&d,NULL,NULL,&e);
  Q=clCreateCommandQueue(c,d,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,&e);
  cl_mem b=clCreateBuffer(c,CL_MEM_READ_WRITE,N*4,NULL,&e);
  int z=7; clEnqueueFillBuffer(Q,b,&z,4,0,N*4,0,NULL,NULL); clFinish(Q);
  void*ptr=clEnqueueMapBuffer(Q,b,CL_FALSE,CL_MAP_READ|CL_MAP_WRITE,0,N*4,0,NULL,&mapev,&e);
  fire=clCreateUserEvent(c,&e);
  clEnqueueUnmapMemObject(Q,b,ptr,1,&fire,&unmap_ev);
  pthread_t t,w; pthread_create(&t,NULL,wk,NULL); pthread_create(&w,NULL,wd,NULL);
  // HOST-SIDE: wait for the unmap to complete (gated on fire=CL_COMPLETE always).
  fprintf(stderr,"[main] clWaitForEvents(unmap)...\n");
  clWaitForEvents(1,&unmap_ev);
  fprintf(stderr,"[main] unmap done; err_flag=%d\n", atomic_load(&err_flag));
  if(atomic_load(&err_flag)){
    fprintf(stderr,"[main] ERROR flag set -> do NOT enqueue downstream read (abort)\n");
    printf("ABORTED cleanly (no downstream)\n"); return 0;
  }
  int*h=malloc(N*4);
  cl_int r=clEnqueueReadBuffer(Q,b,CL_TRUE,0,N*4,h,0,NULL,NULL);
  fprintf(stderr,"[main] read RETURNED %d\n",r); printf("OK read=%d\n",r); return 0;
}

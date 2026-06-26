// Start gate via ONE barrier instead of threading the event into every leaf:
//   start = user event
//   clEnqueueBarrierWithWaitList(Q, [start])   <-- one enqueue, first
//   ... whole graph enqueued normally (all commands wait for the barrier) ...
//   release start; wait last event.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define N 1024
static cl_event fire, proceed, mapev; static cl_command_queue Q;
static int err_path=1;
static void* wk(void*a){(void)a; clWaitForEvents(1,&mapev);
  clSetUserEventStatus(fire, CL_COMPLETE);
  clSetUserEventStatus(proceed, err_path ? -1 : CL_COMPLETE); return NULL;}
static void* wd(void*a){(void)a; sleep(8); fprintf(stderr,"HANG\n"); _exit(42);}
int main(int argc,char**argv){
  for(int i=1;i<argc;i++) if(!strcmp(argv[i],"--success")) err_path=0;
  cl_platform_id p; cl_device_id d; clGetPlatformIDs(1,&p,NULL); clGetDeviceIDs(p,CL_DEVICE_TYPE_DEFAULT,1,&d,NULL);
  char nm[256]={0}; clGetDeviceInfo(d,CL_DEVICE_NAME,256,nm,NULL); fprintf(stderr,"Dev: %s | err=%d\n",nm,err_path);
  cl_int e; cl_context c=clCreateContext(NULL,1,&d,NULL,NULL,&e);
  Q=clCreateCommandQueue(c,d,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,&e);
  cl_mem b=clCreateBuffer(c,CL_MEM_READ_WRITE,N*4,NULL,&e);
  cl_event start=clCreateUserEvent(c,&e);
  // ONE barrier gating everything after it on `start`.
  cl_int be=clEnqueueBarrierWithWaitList(Q,1,&start,NULL);
  fprintf(stderr,"barrier=%d\n",be);
  // Graph enqueued normally — NO start dep needed on individual ops.
  int z=7; clEnqueueFillBuffer(Q,b,&z,4,0,N*4,0,NULL,NULL);
  void*ptr=clEnqueueMapBuffer(Q,b,CL_FALSE,CL_MAP_READ|CL_MAP_WRITE,0,N*4,0,NULL,&mapev,&e);
  fire=clCreateUserEvent(c,&e); proceed=clCreateUserEvent(c,&e);
  cl_event un; clEnqueueUnmapMemObject(Q,b,ptr,1,&fire,&un);
  pthread_t t,w; pthread_create(&t,NULL,wk,NULL); pthread_create(&w,NULL,wd,NULL);
  cl_event wl[2]={un,proceed}; int*h=malloc(N*4); cl_event read_ev;
  cl_int r=clEnqueueReadBuffer(Q,b,CL_FALSE,0,N*4,h,2,wl,&read_ev);
  fprintf(stderr,"[main] graph enqueued (read=%d). Release start.\n",r);
  clSetUserEventStatus(start, CL_COMPLETE);
  cl_int we=clWaitForEvents(1,&read_ev);
  fprintf(stderr,"[main] read_ev wait RETURNED %d\n",we);
  pthread_join(t,NULL);
  printf("done wait=%d\n",we); return 0;
}

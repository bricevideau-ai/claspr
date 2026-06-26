// Brice's design: a START user event gates the FIRST enqueue, released only
// after the WHOLE graph is enqueued — so nothing runs on the device until
// everything is committed, and proceed=-1 can never race the read's enqueue.
//
//   start = user event
//   fill  gated on [start]          (the first enqueue, normally dep-less)
//   map   gated on [fill]
//   fire, proceed = user events
//   unmap gated on [fire]
//   read  (LAST, non-blocking) gated on [unmap, proceed] -> read_ev
//   worker: wait map_ev; fire=CL_COMPLETE; proceed = status (-1 on err)
//   set start = CL_COMPLETE   <-- release the whole graph, AFTER all enqueued
//   clWaitForEvents(read_ev)  <-- wait the last event (NOT clFinish)
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
static void* wk(void*a){(void)a;
  clWaitForEvents(1,&mapev);   // map completes only after `start` released
  clSetUserEventStatus(fire, CL_COMPLETE);
  clSetUserEventStatus(proceed, err_path ? -1 : CL_COMPLETE);
  return NULL;}
static void* wd(void*a){(void)a; sleep(8); fprintf(stderr,"HANG\n"); _exit(42);}
int main(int argc,char**argv){
  for(int i=1;i<argc;i++) if(!strcmp(argv[i],"--success")) err_path=0;
  cl_platform_id p; cl_device_id d; clGetPlatformIDs(1,&p,NULL); clGetDeviceIDs(p,CL_DEVICE_TYPE_DEFAULT,1,&d,NULL);
  char nm[256]={0}; clGetDeviceInfo(d,CL_DEVICE_NAME,256,nm,NULL); fprintf(stderr,"Dev: %s | err=%d\n",nm,err_path);
  cl_int e; cl_context c=clCreateContext(NULL,1,&d,NULL,NULL,&e);
  Q=clCreateCommandQueue(c,d,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,&e);
  cl_mem b=clCreateBuffer(c,CL_MEM_READ_WRITE,N*4,NULL,&e);

  // START gate — gates the first enqueue.
  cl_event start=clCreateUserEvent(c,&e);

  // First enqueue (fill) gated on start (normally would have no deps).
  int z=7; cl_event fillev;
  clEnqueueFillBuffer(Q,b,&z,4,0,N*4,1,&start,&fillev);
  // map gated on fill.
  void*ptr=clEnqueueMapBuffer(Q,b,CL_FALSE,CL_MAP_READ|CL_MAP_WRITE,0,N*4,1,&fillev,&mapev,&e);

  fire=clCreateUserEvent(c,&e); proceed=clCreateUserEvent(c,&e);
  cl_event un; clEnqueueUnmapMemObject(Q,b,ptr,1,&fire,&un);

  pthread_t t,w; pthread_create(&t,NULL,wk,NULL); pthread_create(&w,NULL,wd,NULL);

  // LAST command: non-blocking read gated on [unmap, proceed] -> read_ev.
  cl_event wl[2]={un,proceed}; int*h=malloc(N*4); cl_event read_ev;
  cl_int r=clEnqueueReadBuffer(Q,b,CL_FALSE,0,N*4,h,2,wl,&read_ev);
  fprintf(stderr,"[main] whole graph enqueued (read=%d). Release start.\n",r);

  // Whole graph enqueued -> release it.
  clSetUserEventStatus(start, CL_COMPLETE);

  // Wait the LAST event (not clFinish).
  fprintf(stderr,"[main] clWaitForEvents(read_ev)...\n");
  cl_int we=clWaitForEvents(1,&read_ev);
  fprintf(stderr,"[main] read_ev wait RETURNED %d\n",we);
  // Proper teardown: join the worker only (it may still be in
  // clSetUserEventStatus / holding the handle at process exit). Do NOT clFinish
  // — clFinish on a queue holding a terminated command is the pocl hang/crash we
  // are avoiding; waiting the last event above is the drain.
  pthread_join(t, NULL);
  fprintf(stderr,"[main] worker joined.\n");
  printf("NO HANG wait=%d\n",we); return 0;
}

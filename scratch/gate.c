// Three-event design: fire (unmaps, always CL_COMPLETE), proceed (downstream
// abort), and GATE — the worker waits on GATE before signalling anything, and
// the main thread completes GATE only AFTER the whole graph (incl. the terminal
// non-blocking read) is enqueued. So proceed=-1 can never race the read enqueue.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define N 1024
static cl_event fire, proceed, gate, mapev; static cl_command_queue Q;
static int err_path=1;
static void* wk(void*a){(void)a;
  clWaitForEvents(1,&mapev);            // map ready -> closure could run here
  // closure "runs"; on error we'd stash the slot (host-side flag in real code)
  clWaitForEvents(1,&gate);             // <-- WAIT until graph fully enqueued
  clSetUserEventStatus(fire, CL_COMPLETE);
  clSetUserEventStatus(proceed, err_path ? -1 : CL_COMPLETE);
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
  fire=clCreateUserEvent(c,&e); proceed=clCreateUserEvent(c,&e); gate=clCreateUserEvent(c,&e);
  cl_event un; clEnqueueUnmapMemObject(Q,b,ptr,1,&fire,&un);
  pthread_t t,w; pthread_create(&t,NULL,wk,NULL); pthread_create(&w,NULL,wd,NULL);
  // Enqueue the WHOLE graph (terminal non-blocking read) gated on [unmap, proceed].
  cl_event wl[2]={un,proceed}; int*h=malloc(N*4); cl_event re;
  cl_int r=clEnqueueReadBuffer(Q,b,CL_FALSE,0,N*4,h,2,wl,&re);
  fprintf(stderr,"[main] read enqueued (%d). Now complete GATE.\n",r);
  // Whole graph enqueued -> NOW let the worker signal.
  clSetUserEventStatus(gate, CL_COMPLETE);
  // Terminal wait (drain).
  fprintf(stderr,"[main] clFinish...\n");
  cl_int f=clFinish(Q);
  fprintf(stderr,"[main] clFinish RETURNED %d\n",f);
  printf("NO HANG finish=%d\n",f); return 0;
}

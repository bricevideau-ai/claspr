#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define N 1024
static cl_event A,B,mapev; static cl_command_queue Q;
static void* wk(void*a){(void)a; clWaitForEvents(1,&mapev); // map completes EARLY
  clSetUserEventStatus(A,CL_COMPLETE);
  clSetUserEventStatus(B,-1);   // fire proceed=-1 ASAP — races read enqueue
  return NULL;}
static void* wd(void*a){(void)a; sleep(8); fprintf(stderr,"HANG\n"); _exit(42);}
int main(void){
  cl_platform_id p; cl_device_id d; clGetPlatformIDs(1,&p,NULL); clGetDeviceIDs(p,CL_DEVICE_TYPE_DEFAULT,1,&d,NULL);
  cl_int e; cl_context c=clCreateContext(NULL,1,&d,NULL,NULL,&e);
  Q=clCreateCommandQueue(c,d,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,&e);
  cl_mem b=clCreateBuffer(c,CL_MEM_READ_WRITE,N*4,NULL,&e);
  int z=7; clEnqueueFillBuffer(Q,b,&z,4,0,N*4,0,NULL,NULL); clFinish(Q);
  void*ptr=clEnqueueMapBuffer(Q,b,CL_FALSE,CL_MAP_READ|CL_MAP_WRITE,0,N*4,0,NULL,&mapev,&e);
  A=clCreateUserEvent(c,&e); B=clCreateUserEvent(c,&e);
  cl_event un; clEnqueueUnmapMemObject(Q,b,ptr,1,&A,&un);
  cl_event wl[2]={un,B};
  pthread_t t,w; pthread_create(&t,NULL,wk,NULL); pthread_create(&w,NULL,wd,NULL);
  int*h=malloc(N*4);
  cl_int r=clEnqueueReadBuffer(Q,b,CL_TRUE,0,N*4,h,2,wl,NULL);
  fprintf(stderr,"read RETURNED %d\n",r); return 0;
}

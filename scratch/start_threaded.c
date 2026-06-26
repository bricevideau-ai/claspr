#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define N 1024
static cl_event fire, proceed, mapev, start; static cl_command_queue Q;
static int err_path=1;
static void* wk(void*a){(void)a; clWaitForEvents(1,&mapev);
  clSetUserEventStatus(fire, CL_COMPLETE);
  clSetUserEventStatus(proceed, err_path?-1:CL_COMPLETE); return NULL;}
static void* wd(void*a){(void)a; sleep(8); fprintf(stderr,"HANG\n"); _exit(42);}
// merge start into a wait-list
static int mk(cl_event*out, cl_event* deps, int n){ int k=0; if(start) out[k++]=start; for(int i=0;i<n;i++) out[k++]=deps[i]; return k; }
int main(int argc,char**argv){
  for(int i=1;i<argc;i++) if(!strcmp(argv[i],"--success")) err_path=0;
  cl_platform_id p; cl_device_id d; clGetPlatformIDs(1,&p,NULL); clGetDeviceIDs(p,CL_DEVICE_TYPE_DEFAULT,1,&d,NULL);
  cl_int e; cl_context c=clCreateContext(NULL,1,&d,NULL,NULL,&e);
  Q=clCreateCommandQueue(c,d,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,&e);
  cl_mem b=clCreateBuffer(c,CL_MEM_READ_WRITE,N*4,NULL,&e);
  start=clCreateUserEvent(c,&e);
  cl_event wl[4]; int k;
  // entry leaf fill: wait-list = [start]
  int z=7; cl_event fillev; k=mk(wl,NULL,0); clEnqueueFillBuffer(Q,b,&z,4,0,N*4,k,wl,&fillev);
  // map: deps=[fill] (+start harmlessly)
  cl_event md[1]={fillev}; k=mk(wl,md,1);
  void*ptr=clEnqueueMapBuffer(Q,b,CL_FALSE,CL_MAP_READ|CL_MAP_WRITE,0,N*4,k,wl,&mapev,&e);
  fire=clCreateUserEvent(c,&e); proceed=clCreateUserEvent(c,&e);
  cl_event un; cl_event fd[1]={fire}; k=mk(wl,fd,1); clEnqueueUnmapMemObject(Q,b,ptr,k-1+0? k:k,wl,&un); // unmap gated on [start,fire]
  pthread_t t,w; pthread_create(&t,NULL,wk,NULL); pthread_create(&w,NULL,wd,NULL);
  cl_event rd[2]={un,proceed}; k=mk(wl,rd,2); int*h=malloc(N*4); cl_event re;
  cl_int r=clEnqueueReadBuffer(Q,b,CL_FALSE,0,N*4,h,k,wl,&re);
  fprintf(stderr,"graph enqueued (read=%d). release start.\n",r);
  clSetUserEventStatus(start,CL_COMPLETE);
  cl_int we=clWaitForEvents(1,&re);
  fprintf(stderr,"read wait=%d\n",we);
  pthread_join(t,NULL);
  printf("done=%d\n",we); return 0;
}

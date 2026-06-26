#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#define N 1024
static cl_event U, mapev; static cl_command_queue Q; static int forcehang=1;
static void* wk(void*a){(void)a; clWaitForEvents(1,&mapev);
  if(forcehang) usleep(200000); else usleep(300);
  fprintf(stderr,"[wk] set U=-1\n"); clSetUserEventStatus(U,-1); fprintf(stderr,"[wk] done\n"); return NULL;}
static void* wd(void*a){(void)a; sleep(8); fprintf(stderr,"[wd] HANG\n"); _exit(42);}
int main(int argc,char**argv){
  for(int i=1;i<argc;i++) if(!strcmp(argv[i],"--race")) forcehang=0;
  cl_platform_id p; cl_device_id d; clGetPlatformIDs(1,&p,NULL); clGetDeviceIDs(p,CL_DEVICE_TYPE_DEFAULT,1,&d,NULL);
  char nm[256]={0}; clGetDeviceInfo(d,CL_DEVICE_NAME,256,nm,NULL); fprintf(stderr,"Dev: %s\n",nm);
  cl_int e; cl_context c=clCreateContext(NULL,1,&d,NULL,NULL,&e);
  Q=clCreateCommandQueue(c,d,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,&e);
  cl_mem b=clCreateBuffer(c,CL_MEM_READ_WRITE,N*4,NULL,&e);
  int z=0; clEnqueueFillBuffer(Q,b,&z,4,0,N*4,0,NULL,NULL); clFinish(Q);
  void*ptr=clEnqueueMapBuffer(Q,b,CL_FALSE,CL_MAP_READ|CL_MAP_WRITE,0,N*4,0,NULL,&mapev,&e);
  U=clCreateUserEvent(c,&e);
  cl_event un; clEnqueueUnmapMemObject(Q,b,ptr,1,&U,&un);   // gated unmap, NO defensive unmap
  pthread_t t,w; pthread_create(&t,NULL,wk,NULL); pthread_create(&w,NULL,wd,NULL);
  int*h=malloc(N*4);
  fprintf(stderr,"[main] BLOCKING read gated on gated-unmap...\n");
  cl_int r=clEnqueueReadBuffer(Q,b,CL_TRUE,0,N*4,h,1,&un,NULL);
  fprintf(stderr,"[main] read RETURNED status %d\n",r);
  printf("NO HANG status=%d\n",r); pthread_join(t,NULL); return 0;
}

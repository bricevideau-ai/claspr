// Two-user-event approach:
//   A = "fire unmaps"  : unmap gates on A; ALWAYS completed CL_COMPLETE
//                        (unmap runs cleanly on success AND error — no double
//                         unmap, no map_count corruption).
//   B = "proceed/cancel": downstream read gates on [unmap_ev, B];
//                         B = CL_COMPLETE on success, B = -1 on error (abort).
// Question: on error (B=-1) with the unmap cleanly fired via A, does the
// blocking read hang on NEO, or return an error (abort, no stale data)?
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define N 1024
static cl_event A, B, mapev; static cl_command_queue Q;
static int err_path=1, forcehang=1;
static void* wk(void*a){(void)a; clWaitForEvents(1,&mapev);
  // A always CL_COMPLETE -> unmap fires cleanly (single, correct).
  fprintf(stderr,"[wk] A=CL_COMPLETE (fire unmap)\n"); clSetUserEventStatus(A,CL_COMPLETE);
  if(forcehang) usleep(200000); else usleep(300);
  if(err_path){ fprintf(stderr,"[wk] B=-1 (cancel downstream)\n"); clSetUserEventStatus(B,-1); }
  else        { fprintf(stderr,"[wk] B=CL_COMPLETE (proceed)\n"); clSetUserEventStatus(B,CL_COMPLETE); }
  fprintf(stderr,"[wk] done\n"); return NULL;}
static void* wd(void*a){(void)a; sleep(8); fprintf(stderr,"[wd] HANG\n"); _exit(42);}
int main(int argc,char**argv){
  for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"--success")) err_path=0; if(!strcmp(argv[i],"--race")) forcehang=0; }
  cl_platform_id p; cl_device_id d; clGetPlatformIDs(1,&p,NULL); clGetDeviceIDs(p,CL_DEVICE_TYPE_DEFAULT,1,&d,NULL);
  char nm[256]={0}; clGetDeviceInfo(d,CL_DEVICE_NAME,256,nm,NULL); fprintf(stderr,"Dev: %s | err_path=%d\n",nm,err_path);
  cl_int e; cl_context c=clCreateContext(NULL,1,&d,NULL,NULL,&e);
  Q=clCreateCommandQueue(c,d,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,&e);
  cl_mem b=clCreateBuffer(c,CL_MEM_READ_WRITE,N*4,NULL,&e);
  int z=7; clEnqueueFillBuffer(Q,b,&z,4,0,N*4,0,NULL,NULL); clFinish(Q);
  void*ptr=clEnqueueMapBuffer(Q,b,CL_FALSE,CL_MAP_READ|CL_MAP_WRITE,0,N*4,0,NULL,&mapev,&e);
  A=clCreateUserEvent(c,&e); B=clCreateUserEvent(c,&e);
  cl_event unmap_ev; clEnqueueUnmapMemObject(Q,b,ptr,1,&A,&unmap_ev);   // unmap gated on A
  cl_event waitlist[2]={unmap_ev,B};
  pthread_t t,w; pthread_create(&t,NULL,wk,NULL); pthread_create(&w,NULL,wd,NULL);
  int*h=malloc(N*4);
  fprintf(stderr,"[main] BLOCKING read gated on [unmap_ev, B]...\n");
  cl_int r=clEnqueueReadBuffer(Q,b,CL_TRUE,0,N*4,h,2,waitlist,NULL);
  fprintf(stderr,"[main] read RETURNED status %d (%s)\n",r, r==0?"SUCCESS":"error");
  printf("NO HANG status=%d\n",r); pthread_join(t,NULL); return 0;
}

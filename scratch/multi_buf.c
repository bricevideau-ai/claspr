// Multi-buffer two-event host-seam model.
//   Map M buffers (RW, non-blocking).
//   ONE event A gates ALL M unmaps (A=CL_COMPLETE always -> every unmap fires).
//   ONE event B = proceed/cancel.
//   Downstream blocking read of buffer 0 gates on [ all M unmap events, B ].
//   On error: A=CL_COMPLETE (all unmaps run), B=-1 (downstream aborts).
// Verify: no hang, exactly M unmaps (no double-unmap), read aborts on error.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define N 1024
#define M 4
static cl_event A, B, mapev[M]; static cl_command_queue Q;
static int err_path=1, forcehang=1;
static void* wk(void*a){(void)a;
  clWaitForEvents(M, mapev);
  fprintf(stderr,"[wk] A=CL_COMPLETE (fire all %d unmaps)\n", M); clSetUserEventStatus(A,CL_COMPLETE);
  if(forcehang) usleep(200000); else usleep(300);
  if(err_path){ fprintf(stderr,"[wk] B=-1 (cancel downstream)\n"); clSetUserEventStatus(B,-1); }
  else        { fprintf(stderr,"[wk] B=CL_COMPLETE\n"); clSetUserEventStatus(B,CL_COMPLETE); }
  fprintf(stderr,"[wk] done\n"); return NULL;}
static void* wd(void*a){(void)a; sleep(8); fprintf(stderr,"[wd] HANG\n"); _exit(42);}
int main(int argc,char**argv){
  for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"--success")) err_path=0; if(!strcmp(argv[i],"--race")) forcehang=0; }
  cl_platform_id p; cl_device_id d; clGetPlatformIDs(1,&p,NULL); clGetDeviceIDs(p,CL_DEVICE_TYPE_DEFAULT,1,&d,NULL);
  char nm[256]={0}; clGetDeviceInfo(d,CL_DEVICE_NAME,256,nm,NULL); fprintf(stderr,"Dev: %s | M=%d err_path=%d\n",nm,M,err_path);
  cl_int e; cl_context c=clCreateContext(NULL,1,&d,NULL,NULL,&e);
  Q=clCreateCommandQueue(c,d,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,&e);
  cl_mem buf[M]; void* ptr[M]; int z=7;
  for(int i=0;i<M;i++){ buf[i]=clCreateBuffer(c,CL_MEM_READ_WRITE,N*4,NULL,&e);
    clEnqueueFillBuffer(Q,buf[i],&z,4,0,N*4,0,NULL,NULL); }
  clFinish(Q);
  for(int i=0;i<M;i++) ptr[i]=clEnqueueMapBuffer(Q,buf[i],CL_FALSE,CL_MAP_READ|CL_MAP_WRITE,0,N*4,0,NULL,&mapev[i],&e);
  A=clCreateUserEvent(c,&e); B=clCreateUserEvent(c,&e);
  cl_event unmap_ev[M];
  for(int i=0;i<M;i++) clEnqueueUnmapMemObject(Q,buf[i],ptr[i],1,&A,&unmap_ev[i]); // all gate on A
  // downstream read of buf[0] gated on [ all M unmaps, B ]
  cl_event wl[M+1]; for(int i=0;i<M;i++) wl[i]=unmap_ev[i]; wl[M]=B;
  pthread_t t,w; pthread_create(&t,NULL,wk,NULL); pthread_create(&w,NULL,wd,NULL);
  int*h=malloc(N*4);
  fprintf(stderr,"[main] BLOCKING read gated on [%d unmaps + B]...\n", M);
  cl_int r=clEnqueueReadBuffer(Q,buf[0],CL_TRUE,0,N*4,h,M+1,wl,NULL);
  fprintf(stderr,"[main] read RETURNED status %d (%s)\n",r, r==0?"SUCCESS":"error");
  printf("NO HANG status=%d\n",r); pthread_join(t,NULL); return 0;
}

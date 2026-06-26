// Concurrent release race WITH a registered completion callback (the missing
// ingredient): pocl's async-callback thread releases its callback-window retain
// concurrently with the app releasing its own reference -> clReleaseEvent races
// itself at the free.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#define PAIRS 64
#define ITERS 20000
static cl_context ctx;
static void CL_CALLBACK cb(cl_event e, cl_int s, void* u){ (void)e;(void)s;(void)u; }
int main(void){
  cl_platform_id p; cl_device_id d; clGetPlatformIDs(1,&p,NULL); clGetDeviceIDs(p,CL_DEVICE_TYPE_DEFAULT,1,&d,NULL);
  char nm[256]={0}; clGetDeviceInfo(d,CL_DEVICE_NAME,256,nm,NULL); fprintf(stderr,"Dev: %s\n",nm);
  cl_int e; ctx=clCreateContext(NULL,1,&d,NULL,NULL,&e);
  for(int it=0; it<ITERS; it++){
    cl_event ev[PAIRS];
    for(int i=0;i<PAIRS;i++){
      ev[i]=clCreateUserEvent(ctx,&e);
      // register CL_COMPLETE callback (app keeps its 1 reference)
      clSetEventCallback(ev[i], CL_COMPLETE, cb, NULL);
    }
    // Complete all -> pocl async-callback thread fires cb + does its
    // callback-window release. CONCURRENTLY, main releases the app reference.
    for(int i=0;i<PAIRS;i++) clSetUserEventStatus(ev[i], -1); // negative = error path
    for(int i=0;i<PAIRS;i++) clReleaseEvent(ev[i]);            // app release, racing the cb thread
    if((it&0x3ff)==0) fprintf(stderr,"iter %d ok\n",it);
  }
  fprintf(stderr,"survived %d\n",ITERS); printf("NO RACE\n"); return 0;
}

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1024
int main(void){
  cl_platform_id p; cl_device_id d; clGetPlatformIDs(1,&p,NULL); clGetDeviceIDs(p,CL_DEVICE_TYPE_DEFAULT,1,&d,NULL);
  char nm[256]={0}; clGetDeviceInfo(d,CL_DEVICE_NAME,256,nm,NULL); fprintf(stderr,"Dev: %s\n",nm);
  cl_int e; cl_context c=clCreateContext(NULL,1,&d,NULL,NULL,&e);
  cl_command_queue q=clCreateCommandQueue(c,d,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,&e);
  cl_mem b=clCreateBuffer(c,CL_MEM_READ_WRITE,N*4,NULL,&e);
  int z=0; clEnqueueFillBuffer(q,b,&z,4,0,N*4,0,NULL,NULL); clFinish(q);
  void*ptr=clEnqueueMapBuffer(q,b,CL_TRUE,CL_MAP_READ|CL_MAP_WRITE,0,N*4,0,NULL,NULL,&e);
  fprintf(stderr,"mapped (map_count now 1)\n");
  cl_event U=clCreateUserEvent(c,&e);
  cl_event un; cl_int ue=clEnqueueUnmapMemObject(q,b,ptr,1,&U,&un);
  fprintf(stderr,"enqueue gated unmap -> %d (map_count decremented at ENQUEUE on NEO)\n",ue);
  // Before cancel: try a second unmap NOW (this is what claspr does).
  cl_int d1=clEnqueueUnmapMemObject(q,b,ptr,0,NULL,NULL);
  fprintf(stderr,"2nd unmap BEFORE cancel -> %d %s\n",d1, d1==-30?"(CL_INVALID_VALUE)":"");
  // Now cancel.
  clSetUserEventStatus(U,-1);
  fprintf(stderr,"cancelled U=-1\n");
  // After cancel: try unmap again.
  cl_int d2=clEnqueueUnmapMemObject(q,b,ptr,0,NULL,NULL);
  fprintf(stderr,"unmap AFTER cancel -> %d %s\n",d2, d2==-30?"(CL_INVALID_VALUE)":"");
  return 0;
}

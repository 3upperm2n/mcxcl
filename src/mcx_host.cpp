/*******************************************************************************
**
**  Monte Carlo eXtreme (MCX)  - GPU accelerated Monte Carlo 3D photon migration
**      -- OpenCL edition
**  Author: Qianqian Fang <fangq at nmr.mgh.harvard.edu>
**
**  Reference (Fang2009):
**        Qianqian Fang and David A. Boas, "Monte Carlo Simulation of Photon 
**        Migration in 3D Turbid Media Accelerated by Graphics Processing 
**        Units," Optics Express, vol. 17, issue 22, pp. 20178-20190 (2009)
**
**  mcx_host.cpp: Host code for OpenCL
**
**  Unpublished work, see LICENSE.txt for details
**
*******************************************************************************/

#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <math.h>
#include <unistd.h>
#include "mcx_host.hpp"
#include "tictoc.h"
#include "mcx_const.h"

#include <pthread.h>
#define NUM_THREADS 4

extern cl_event kernelevent;


char *print_cl_errstring(cl_int err) {
    switch (err) {
        case CL_SUCCESS:                          return strdup("Success!");
        case CL_DEVICE_NOT_FOUND:                 return strdup("Device not found.");
        case CL_DEVICE_NOT_AVAILABLE:             return strdup("Device not available");
        case CL_COMPILER_NOT_AVAILABLE:           return strdup("Compiler not available");
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return strdup("Memory object allocation failure");
        case CL_OUT_OF_RESOURCES:                 return strdup("Out of resources");
        case CL_OUT_OF_HOST_MEMORY:               return strdup("Out of host memory");
        case CL_PROFILING_INFO_NOT_AVAILABLE:     return strdup("Profiling information not available");
        case CL_MEM_COPY_OVERLAP:                 return strdup("Memory copy overlap");
        case CL_IMAGE_FORMAT_MISMATCH:            return strdup("Image format mismatch");
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:       return strdup("Image format not supported");
        case CL_BUILD_PROGRAM_FAILURE:            return strdup("Program build failure");
        case CL_MAP_FAILURE:                      return strdup("Map failure");
        case CL_INVALID_VALUE:                    return strdup("Invalid value");
        case CL_INVALID_DEVICE_TYPE:              return strdup("Invalid device type");
        case CL_INVALID_PLATFORM:                 return strdup("Invalid platform");
        case CL_INVALID_DEVICE:                   return strdup("Invalid device");
        case CL_INVALID_CONTEXT:                  return strdup("Invalid context");
        case CL_INVALID_QUEUE_PROPERTIES:         return strdup("Invalid queue properties");
        case CL_INVALID_COMMAND_QUEUE:            return strdup("Invalid command queue");
        case CL_INVALID_HOST_PTR:                 return strdup("Invalid host pointer");
        case CL_INVALID_MEM_OBJECT:               return strdup("Invalid memory object");
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  return strdup("Invalid image format descriptor");
        case CL_INVALID_IMAGE_SIZE:               return strdup("Invalid image size");
        case CL_INVALID_SAMPLER:                  return strdup("Invalid sampler");
        case CL_INVALID_BINARY:                   return strdup("Invalid binary");
        case CL_INVALID_BUILD_OPTIONS:            return strdup("Invalid build options");
        case CL_INVALID_PROGRAM:                  return strdup("Invalid program");
        case CL_INVALID_PROGRAM_EXECUTABLE:       return strdup("Invalid program executable");
        case CL_INVALID_KERNEL_NAME:              return strdup("Invalid kernel name");
        case CL_INVALID_KERNEL_DEFINITION:        return strdup("Invalid kernel definition");
        case CL_INVALID_KERNEL:                   return strdup("Invalid kernel");
        case CL_INVALID_ARG_INDEX:                return strdup("Invalid argument index");
        case CL_INVALID_ARG_VALUE:                return strdup("Invalid argument value");
        case CL_INVALID_ARG_SIZE:                 return strdup("Invalid argument size");
        case CL_INVALID_KERNEL_ARGS:              return strdup("Invalid kernel arguments");
        case CL_INVALID_WORK_DIMENSION:           return strdup("Invalid work dimension");
        case CL_INVALID_WORK_GROUP_SIZE:          return strdup("Invalid work group size");
        case CL_INVALID_WORK_ITEM_SIZE:           return strdup("Invalid work item size");
        case CL_INVALID_GLOBAL_OFFSET:            return strdup("Invalid global offset");
        case CL_INVALID_EVENT_WAIT_LIST:          return strdup("Invalid event wait list");
        case CL_INVALID_EVENT:                    return strdup("Invalid event");
        case CL_INVALID_OPERATION:                return strdup("Invalid operation");
        case CL_INVALID_GL_OBJECT:                return strdup("Invalid OpenGL object");
        case CL_INVALID_BUFFER_SIZE:              return strdup("Invalid buffer size");
        case CL_INVALID_MIP_LEVEL:                return strdup("Invalid mip-map level");
        default:                                  return strdup("Unknown");
    }
}

struct kernel_thread_data
{
	int	thread_id;
	cl_command_queue mcxqueue;
	cl_mem gparam;
	MCXParam param;
	cl_kernel mcxkernel;
	size_t *mcgrid;
	size_t *mcblock;
	cl_mem gdetected;
	int detected;
};

/*
   kernel execution function per thread
*/
void *Kernel_Exe(void *threadarg)
{
	int taskid;
	cl_command_queue mcxqueue;
	cl_mem gparam;
	MCXParam param;
	cl_kernel mcxkernel;
	size_t *mcgrid;
	size_t *mcblock;
	cl_mem gdetected;
	int detected;

	struct kernel_thread_data *my_data;

	my_data = (struct kernel_thread_data *) threadarg;
	taskid = my_data->thread_id;
	mcxqueue = my_data->mcxqueue;
	gparam = my_data->gparam;
	param = my_data->param;
	mcxkernel = my_data->mcxkernel;
	mcgrid = my_data->mcgrid;
	mcblock = my_data->mcblock;
	gdetected = my_data->gdetected;
	detected = my_data->detected;

	printf("Thread %d: mcgrid: %ld  mcblock: %ld\n", taskid, *mcgrid, *mcblock);
	//printf("Thread %d: mcxqueue: %x  mcxkernel: %x\n", taskid, mcxqueue, mcxkernel);

	OCL_ASSERT((clEnqueueWriteBuffer(mcxqueue, gparam, CL_TRUE, 0, sizeof(MCXParam), &param, 0, NULL, NULL)));
	OCL_ASSERT((clSetKernelArg(mcxkernel, 12, sizeof(cl_mem), (void*)&gparam)));
	OCL_ASSERT((clEnqueueNDRangeKernel(mcxqueue, mcxkernel, 1, NULL, mcgrid, mcblock, 0, NULL, NULL)));
	//clFlush(mcxqueue);
	//
	OCL_ASSERT((clEnqueueReadBuffer(mcxqueue, gdetected, CL_TRUE, 0, sizeof(uint), &detected, 0, NULL, NULL)));

	pthread_exit(NULL);
}

struct kernel_thread_data kernel_thread_data_array[NUM_THREADS];


/*
   assert cuda memory allocation result
*/
void ocl_assess(int cuerr,const char *file,const int linenum){
     if(cuerr!=CL_SUCCESS){
         mcx_error(-(int)cuerr,print_cl_errstring(cuerr),file,linenum);
     }
}


/*
  query GPU info and set active GPU
*/
cl_platform_id mcx_list_gpu(Config *cfg,unsigned int *activedev,cl_device_id *activedevlist){

    uint i,j,k,cuid=0,devnum;
    cl_uint numPlatforms,devparam,clockspeed;
    cl_ulong devmem,constmem;
    cl_platform_id platform = NULL, activeplatform=NULL;
    cl_device_type devtype[]={CL_DEVICE_TYPE_GPU,CL_DEVICE_TYPE_CPU};
    cl_context context;                 // compute context
    const char *devname[]={"GPU","CPU"};
    char pbuf[100];
    cl_context_properties cps[3]={CL_CONTEXT_PLATFORM, 0, 0};
    cl_int status = 0;
    size_t deviceListSize;

    clGetPlatformIDs(0, NULL, &numPlatforms);
    if(activedev) *activedev=0;

    if (numPlatforms>0) {
        cl_platform_id* platforms =(cl_platform_id*)malloc(sizeof(cl_platform_id)*numPlatforms);
        OCL_ASSERT((clGetPlatformIDs(numPlatforms, platforms, NULL)));
        for (i = 0; i < numPlatforms; ++i) {
            platform = platforms[i];
	    if(1){
                OCL_ASSERT((clGetPlatformInfo(platforms[i],
                          CL_PLATFORM_NAME,sizeof(pbuf),pbuf,NULL)));
	        if(cfg->isgpuinfo) printf("Platform [%d] Name %s\n",i,pbuf);
                cps[1]=(cl_context_properties)platform;
        	for(j=0; j<2; j++){
		    cl_device_id * devices;
		    context=clCreateContextFromType(cps,devtype[j],NULL,NULL,&status);
		    if(status!=CL_SUCCESS){
		            clReleaseContext(context);
			    continue;
		    }
		    OCL_ASSERT((clGetContextInfo(context, CL_CONTEXT_DEVICES,0,NULL,&deviceListSize)));
                    devices = (cl_device_id*)malloc(deviceListSize);
                    OCL_ASSERT((clGetContextInfo(context,CL_CONTEXT_DEVICES,deviceListSize,devices,NULL)));
		    devnum=deviceListSize/sizeof(cl_device_id);
                    for(k=0;k<devnum;k++){
		         if(cfg->deviceid[cuid++]=='1'){
				activedevlist[(*activedev)++]=devices[k];
				if(activeplatform && activeplatform!=platform){
					fprintf(stderr,"Error: one can not mix devices between different platforms\n");
					exit(-1);
				}
				activeplatform=platform;
                          }
			  if(cfg->isgpuinfo){
        	        	OCL_ASSERT((clGetDeviceInfo(devices[k],CL_DEVICE_NAME,100,(void*)&pbuf,NULL)));
                		printf("============ %s device ID %d [%d of %d]: %s  ============\n",devname[j],cuid,k+1,devnum,pbuf);
				OCL_ASSERT((clGetDeviceInfo(devices[k],CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),(void*)&devparam,NULL)));
                		OCL_ASSERT((clGetDeviceInfo(devices[k],CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(cl_ulong),(void*)&devmem,NULL)));
                		printf(" Compute units   :\t%d core(s)\n",(uint)devparam);
                		printf(" Global memory   :\t%ld B\n",(unsigned long)devmem);
                		OCL_ASSERT((clGetDeviceInfo(devices[k],CL_DEVICE_LOCAL_MEM_SIZE,sizeof(cl_ulong),(void*)&devmem,NULL)));
                		printf(" Local memory    :\t%ld B\n",(unsigned long)devmem);
                		OCL_ASSERT((clGetDeviceInfo(devices[k],CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,sizeof(cl_ulong),(void*)&constmem,NULL)));
                		printf(" Constant memory :\t%ld B\n",(unsigned long)constmem);
                		OCL_ASSERT((clGetDeviceInfo(devices[k],CL_DEVICE_MAX_CLOCK_FREQUENCY,sizeof(cl_uint),(void*)&clockspeed,NULL)));
                		printf(" Clock speed     :\t%d MHz\n",clockspeed);
                      	  }
                    }
                    free(devices);
                    clReleaseContext(context);
               }
	    }
        }
        free(platforms);
    }
    if(cfg->isgpuinfo==2) exit(0);
    return activeplatform;
}


/*
   master driver code to run MC simulations
*/
void mcx_run_simulation(Config *cfg,float *fluence,float *totalenergy){

	cl_uint i,j,iter;
	cl_float  minstep=MIN(MIN(cfg->steps.x,cfg->steps.y),cfg->steps.z);
	cl_float t,twindow0,twindow1;
	cl_float fullload=0.f;
	cl_float *energy;
	cl_int stopsign=0;
	cl_uint detected=0,workdev;

	cl_uint tic,tic0,tic1,toc=0,fieldlen;
	cl_uint4 cp0={{cfg->crop0.x,cfg->crop0.y,cfg->crop0.z,cfg->crop0.w}};
	cl_uint4 cp1={{cfg->crop1.x,cfg->crop1.y,cfg->crop1.z,cfg->crop1.w}};
	cl_uint2 cachebox;
	cl_uint4 dimlen;

	cl_context *mcxcontext;
	cl_command_queue *mcxqueue;          // compute command queue
	cl_program *mcxprogram;                 // compute mcxprogram
	cl_kernel *mcxkernel;                   // compute mcxkernel


	pthread_t threads[NUM_THREADS];
	pthread_attr_t attr;
	int rc;
	void *pstatus;


	cl_int status = 0;
	cl_device_id devices[MAX_DEVICE];
	cl_event * waittoread;
	cl_platform_id platform = NULL;

	cl_uint *cucount,totalcucore;
	cl_uint  devid=0;

	//cl_mem gmedia,gproperty,gparam;
	cl_mem *gmedia, *gproperty, *gparam;
	cl_mem *gfield,*gdetphoton,*gseed,*genergy;
	cl_mem *gstopsign,*gdetected,*gdetpos;

	size_t mcgrid[1], mcblock[1];

	cl_uint dimxyz=cfg->dim.x*cfg->dim.y*cfg->dim.z;

	cl_uchar  *media=(cl_uchar *)(cfg->vol);
	cl_float  *field;

	cl_uint   *Pseed;
	float  *Pdet;
	char opt[MAX_PATH_LENGTH]={'\0'};
	cl_uint detreclen=cfg->medianum+1;

	MCXParam param={{{cfg->srcpos.x,cfg->srcpos.y,cfg->srcpos.z,1.f}},
		{{cfg->srcdir.x,cfg->srcdir.y,cfg->srcdir.z,0.f}},
		{{(float)cfg->dim.x,(float)cfg->dim.y,(float)cfg->dim.z,0}},dimlen,cp0,cp1,cachebox,
		minstep,0.f,0.f,cfg->tend,R_C0*cfg->unitinmm,(uint)cfg->isrowmajor,
		(uint)cfg->issave2pt,(uint)cfg->isreflect,(uint)cfg->isrefint,(uint)cfg->issavedet,1.f/cfg->tstep,
		cfg->minenergy,
		cfg->sradius*cfg->sradius,minstep*R_C0*cfg->unitinmm,cfg->maxdetphoton,
		cfg->medianum-1,cfg->detnum,0,0};

	platform=mcx_list_gpu(cfg,&workdev,devices);

	if(workdev>MAX_DEVICE)
		workdev=MAX_DEVICE;

	if(devices == NULL){
		OCL_ASSERT(-1);
	}

	cl_context_properties cps[3]={CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};

	/* Use NULL for backward compatibility */
	cl_context_properties* cprops=(platform==NULL)?NULL:cps;
	//OCL_ASSERT(((mcxcontext=clCreateContextFromType(cprops,CL_DEVICE_TYPE_ALL,NULL,NULL,&status),status)));


	// start pthread
	mcxcontext = (cl_context *) malloc(workdev*sizeof(cl_context));
	mcxprogram = (cl_program *)malloc(workdev*sizeof(cl_program));
	mcxkernel = (cl_kernel*)malloc(workdev*sizeof(cl_kernel));

	gmedia=(cl_mem *)malloc(workdev*sizeof(cl_mem));
	gproperty=(cl_mem *)malloc(workdev*sizeof(cl_mem));
	gparam=(cl_mem *)malloc(workdev*sizeof(cl_mem));
	// end pthread



	mcxqueue= (cl_command_queue*)malloc(workdev*sizeof(cl_command_queue));
	waittoread=(cl_event *)malloc(workdev*sizeof(cl_event));
	cucount=(cl_uint *)calloc(workdev,sizeof(cl_uint));

	gseed=(cl_mem *)malloc(workdev*sizeof(cl_mem));
	gfield=(cl_mem *)malloc(workdev*sizeof(cl_mem));
	gdetphoton=(cl_mem *)malloc(workdev*sizeof(cl_mem));
	genergy=(cl_mem *)malloc(workdev*sizeof(cl_mem));
	gstopsign=(cl_mem *)malloc(workdev*sizeof(cl_mem));
	gdetected=(cl_mem *)malloc(workdev*sizeof(cl_mem));
	gdetpos=(cl_mem *)malloc(workdev*sizeof(cl_mem));

	/* The block is to move the declaration of prop closer to its use */
	cl_command_queue_properties prop = CL_QUEUE_PROFILING_ENABLE;

	totalcucore=0;
	for(i=0;i<workdev;i++){
		char pbuf[100]={'\0'};
		OCL_ASSERT(((mcxcontext[i]=clCreateContextFromType(cprops,CL_DEVICE_TYPE_ALL,NULL,NULL,&status),status)));

		//OCL_ASSERT(((mcxqueue[i]=clCreateCommandQueue(mcxcontext,devices[i],prop,&status),status)));
		OCL_ASSERT(((mcxqueue[i]=clCreateCommandQueue(mcxcontext[i],devices[i],prop,&status),status)));

		OCL_ASSERT((clGetDeviceInfo(devices[i],CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),(void*)(cucount+i),NULL)));
		OCL_ASSERT((clGetDeviceInfo(devices[i],CL_DEVICE_NAME,100,(void*)&pbuf,NULL)));

		printf("=> device %d: %s\n", i, pbuf);

		if(strstr(pbuf,"ATI")){
			cucount[i]*=(80/5); // an ati core typically has 80 SP, and 80/5=16 VLIW
		}else if(strstr(pbuf,"GeForce") || strstr(pbuf,"Quadro") || strstr(pbuf,"Tesla")){
			cucount[i]*=8;  // an nvidia MP typically has 8 SP
		}
		totalcucore+=cucount[i];
	}
	fullload=0.f;
	for(i=0;i<workdev;i++)
		fullload+=cfg->workload[i];

	if(fullload<EPS){
		for(i=0;i<workdev;i++)
			cfg->workload[i]=cucount[i];
		fullload=totalcucore;
	}

	if(cfg->respin>1){
		field=(cl_float *)calloc(sizeof(cl_float)*dimxyz,cfg->maxgate*2); //the second half will be used to accumul$
	}else{
		field=(cl_float *)calloc(sizeof(cl_float)*dimxyz,cfg->maxgate);
	}
	if(cfg->nthread%cfg->nblocksize)
		cfg->nthread=(cfg->nthread/cfg->nblocksize)*cfg->nblocksize;

	mcgrid[0]=cfg->nthread;
	mcblock[0]=cfg->nblocksize;

	Pseed=(cl_uint*)malloc(sizeof(cl_uint)*cfg->nthread*RAND_SEED_LEN);
	energy=(cl_float*)calloc(sizeof(cl_float),cfg->nthread*3);
	Pdet=(float*)calloc(cfg->maxdetphoton,sizeof(float)*(cfg->medianum+1));

	cachebox.x=(cp1.x-cp0.x+1);
	cachebox.y=(cp1.y-cp0.y+1)*(cp1.x-cp0.x+1);
	dimlen.x=cfg->dim.x;
	dimlen.y=cfg->dim.x*cfg->dim.y;
	dimlen.z=cfg->dim.x*cfg->dim.y*cfg->dim.z;

	memcpy(&(param.dimlen.x),&(dimlen.x),sizeof(uint4));
	memcpy(&(param.cachebox.x),&(cachebox.x),sizeof(uint2));
	param.idx1dorig=(int(floorf(param.ps.z))*dimlen.y+
			int(floorf(param.ps.y))*dimlen.x+
			int(floorf(param.ps.x)));
	param.mediaidorig=(cfg->vol[param.idx1dorig] & MED_MASK);

	if(cfg->seed>0)
		srand(cfg->seed);
	else
		srand(time(0));

	//OCL_ASSERT(((gmedia=clCreateBuffer(mcxcontext,RO_MEM, sizeof(cl_uchar)*(dimxyz),media,&status),status)));
	//OCL_ASSERT(((gproperty=clCreateBuffer(mcxcontext,RO_MEM, cfg->medianum*sizeof(Medium),cfg->prop,&status),status)));
	//OCL_ASSERT(((gparam=clCreateBuffer(mcxcontext,RO_MEM, sizeof(MCXParam),&param,&status),status)));

	for(i=0;i<workdev;i++){
		for (j=0; j<cfg->nthread*RAND_SEED_LEN;j++)
			Pseed[j]=rand();

		// pthread
		OCL_ASSERT(((gmedia[i]=clCreateBuffer(mcxcontext[i],RO_MEM, sizeof(cl_uchar)*(dimxyz),media,&status),status)));
		OCL_ASSERT(((gproperty[i]=clCreateBuffer(mcxcontext[i],RO_MEM, cfg->medianum*sizeof(Medium),cfg->prop,&status),status)));
		OCL_ASSERT(((gparam[i]=clCreateBuffer(mcxcontext[i],RO_MEM, sizeof(MCXParam),&param,&status),status)));

		OCL_ASSERT(((gseed[i]=clCreateBuffer(mcxcontext[i],RW_MEM, sizeof(cl_uint)*cfg->nthread*RAND_SEED_LEN,Pseed,&status),status)));
		OCL_ASSERT(((gfield[i]=clCreateBuffer(mcxcontext[i],RW_MEM, sizeof(cl_float)*(dimxyz)*cfg->maxgate,field,&status),status)));
		OCL_ASSERT(((gdetphoton[i]=clCreateBuffer(mcxcontext[i],RW_MEM, sizeof(float)*cfg->maxdetphoton*(cfg->medianum+1),Pdet,&status),status)));
		OCL_ASSERT(((genergy[i]=clCreateBuffer(mcxcontext[i],RW_MEM, sizeof(float)*(cfg->nthread<<1),energy,&status),status)));
		OCL_ASSERT(((gstopsign[i]=clCreateBuffer(mcxcontext[i],RW_PTR, sizeof(cl_uint),&stopsign,&status),status)));
		OCL_ASSERT(((gdetected[i]=clCreateBuffer(mcxcontext[i],RW_MEM, sizeof(cl_uint),&detected,&status),status)));
		OCL_ASSERT(((gdetpos[i]=clCreateBuffer(mcxcontext[i],RO_MEM, cfg->detnum*sizeof(float4),cfg->detpos,&status),status)));
	}

	fprintf(cfg->flog,"\
			===============================================================================\n\
			=                     Monte Carlo eXtreme (MCX) -- OpenCL                     =\n\
			=           Copyright (c) 2009-2016 Qianqian Fang <q.fang at neu.edu>         =\n\
			=                                                                             =\n\
			=                    Computational Imaging Laboratory (CIL)                   =\n\
			=             Department of Bioengineering, Northeastern University           =\n\
			===============================================================================\n\
			$MCXCL$Rev::    $ Last Commit $Date::                     $ by $Author:: fangq$\n\
			===============================================================================\n");

	tic=StartTimer();
	if(cfg->issavedet)
		fprintf(cfg->flog,"- variant name: [%s] compiled with OpenCL version [%d]\n",
				"Detective MCXCL",CL_VERSION_1_0);
	else
		fprintf(cfg->flog,"- code name: [Vanilla MCXCL] compiled with OpenCL version [%d]\n",
				CL_VERSION_1_0);

	fprintf(cfg->flog,"- compiled with: [RNG] %s [Seed Length] %d\n",MCX_RNG_NAME,RAND_SEED_LEN);
	fprintf(cfg->flog,"initializing streams ...\t");
	fflush(cfg->flog);
	fieldlen=dimxyz*cfg->maxgate;

	fprintf(cfg->flog,"init complete : %d ms\n",GetTimeMillis()-tic);

	//OCL_ASSERT(((mcxprogram=clCreateProgramWithSource(mcxcontext, 1,(const char **)&(cfg->clsource), NULL, &status),status)));
	for(i=0;i<workdev;i++){
		OCL_ASSERT(((mcxprogram[i]=clCreateProgramWithSource(mcxcontext[i], 1,(const char **)&(cfg->clsource), NULL, &status),status)));
	}

	sprintf(opt,"-cl-mad-enable -cl-fast-relaxed-math %s",cfg->compileropt);
	if(cfg->issavedet)
		sprintf(opt+strlen(opt)," -D MCX_SAVE_DETECTORS");
	if(cfg->isreflect)
		sprintf(opt+strlen(opt)," -D MCX_DO_REFLECTION");
	sprintf(opt+strlen(opt)," %s",cfg->compileropt);


	for(i=0;i<workdev;i++){
		status=clBuildProgram(mcxprogram[i], 0, NULL, opt, NULL, NULL);

		if(status!=CL_SUCCESS){
			size_t len;
			char *msg;
			// get the details on the error, and store it in buffer
			clGetProgramBuildInfo(mcxprogram[i],devices[i],CL_PROGRAM_BUILD_LOG,0,NULL,&len); 
			msg=new char[len];
			clGetProgramBuildInfo(mcxprogram[i],devices[i],CL_PROGRAM_BUILD_LOG,len,msg,NULL); 
			fprintf(cfg->flog,"Kernel build error:\n%s\n", msg);
			mcx_error(-(int)status,(char*)("Error: Failed to build program executable!"),__FILE__,__LINE__);
			delete msg;
		}
	}

	fprintf(cfg->flog,"build program complete (all the devices) : %d ms\n",GetTimeMillis()-tic);

	//mcxkernel=(cl_kernel*)malloc(workdev*sizeof(cl_kernel));

	for(i=0;i<workdev;i++){
		cl_int threadphoton, oddphotons;

		threadphoton=(int)(cfg->nphoton*cfg->workload[i]/(fullload*cfg->nthread*cfg->respin));
		oddphotons=(int)(cfg->nphoton*cfg->workload[i]/(fullload*cfg->respin)-threadphoton*cfg->nthread);

		fprintf(cfg->flog,"- [device %d] threadph=%d oddphotons=%d np=%.1f nthread=%d repetition=%d\n",i,threadphoton,oddphotons,
				cfg->nphoton*cfg->workload[i]/fullload,cfg->nthread,cfg->respin);

		OCL_ASSERT(((mcxkernel[i] = clCreateKernel(mcxprogram[i], "mcx_main_loop", &status),status)));

		OCL_ASSERT((clSetKernelArg(mcxkernel[i], 0, sizeof(cl_uint),(void*)&threadphoton)));
		OCL_ASSERT((clSetKernelArg(mcxkernel[i], 1, sizeof(cl_uint),(void*)&oddphotons)));
		//OCL_ASSERT((clSetKernelArg(mcxkernel[i], 2, sizeof(cl_mem), (void*)&gmedia)));
		OCL_ASSERT((clSetKernelArg(mcxkernel[i], 2, sizeof(cl_mem), (void*)(gmedia+i))));
		OCL_ASSERT((clSetKernelArg(mcxkernel[i], 3, sizeof(cl_mem), (void*)(gfield+i))));
		OCL_ASSERT((clSetKernelArg(mcxkernel[i], 4, sizeof(cl_mem), (void*)(genergy+i))));
		OCL_ASSERT((clSetKernelArg(mcxkernel[i], 5, sizeof(cl_mem), (void*)(gseed+i))));
		OCL_ASSERT((clSetKernelArg(mcxkernel[i], 6, sizeof(cl_mem), (void*)(gdetphoton+i))));
		//OCL_ASSERT((clSetKernelArg(mcxkernel[i], 7, sizeof(cl_mem), (void*)&gproperty)));
		OCL_ASSERT((clSetKernelArg(mcxkernel[i], 7, sizeof(cl_mem), (void*)(gproperty+i))));
		OCL_ASSERT((clSetKernelArg(mcxkernel[i], 8, sizeof(cl_mem), (void*)(gdetpos+i))));
		OCL_ASSERT((clSetKernelArg(mcxkernel[i], 9, sizeof(cl_mem), (void*)(gstopsign+i))));
		OCL_ASSERT((clSetKernelArg(mcxkernel[i],10, sizeof(cl_mem), (void*)(gdetected+i))));
		OCL_ASSERT((clSetKernelArg(mcxkernel[i],11, cfg->issavedet? sizeof(cl_float)*cfg->nblocksize*param.maxmedia : 1, NULL)));
	}
	fprintf(cfg->flog,"set kernel arguments complete : %d ms\n",GetTimeMillis()-tic);

	if(cfg->exportfield==NULL)
		cfg->exportfield=(float *)calloc(sizeof(float)*cfg->dim.x*cfg->dim.y*cfg->dim.z,cfg->maxgate*2);
	if(cfg->exportdetected==NULL)
		cfg->exportdetected=(float*)malloc((cfg->medianum+1)*cfg->maxdetphoton*sizeof(float));

	cfg->energytot=0.f;
	cfg->energyesc=0.f;
	cfg->runtime=0;

	//simulate for all time-gates in maxgate groups per run

	cl_float Vvox;
	Vvox=cfg->steps.x*cfg->steps.y*cfg->steps.z;
	tic0=GetTimeMillis();

	cl_uint tic_pthread;

	for(t=cfg->tstart;t<cfg->tend;t+=cfg->tstep*cfg->maxgate){
		twindow0=t;
		twindow1=t+cfg->tstep*cfg->maxgate;

		fprintf(cfg->flog,"lauching mcx_main_loop for time window [%.1fns %.1fns] ...\n"
				,twindow0*1e9,twindow1*1e9);

		//total number of repetition for the simulations, results will be accumulated to field
		for(iter=0;iter<cfg->respin;iter++){
			fprintf(cfg->flog,"simulation run#%2d ... \t",iter+1); fflush(cfg->flog);
			param.twin0=twindow0;
			param.twin1=twindow1;

			/*
			for(devid=0;devid<workdev;devid++){
				OCL_ASSERT((clEnqueueWriteBuffer(mcxqueue[devid],gparam,CL_TRUE,0,sizeof(MCXParam),&param, 0, NULL, NULL)));
				OCL_ASSERT((clSetKernelArg(mcxkernel[devid],12, sizeof(cl_mem), (void*)&gparam)));
				// launch mcxkernel
#ifndef USE_OS_TIMER
				OCL_ASSERT((clEnqueueNDRangeKernel(mcxqueue[devid],mcxkernel[devid],1,NULL,mcgrid,mcblock, 0, NULL, &kernelevent)));
#else
				OCL_ASSERT((clEnqueueNDRangeKernel(mcxqueue[devid],mcxkernel[devid],1,NULL,mcgrid,mcblock, 0, NULL, NULL)));
#endif
				OCL_ASSERT((clEnqueueReadBuffer(mcxqueue[devid],gdetected[devid],CL_FALSE,0,sizeof(uint),
								&detected, 0, NULL, waittoread+devid)));
			}
*/

			tic_pthread=GetTimeMillis();   

			pthread_attr_init(&attr);
			pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

			for(cl_uint devid=0;devid<workdev;devid++){

				kernel_thread_data_array[devid].thread_id = devid;
				kernel_thread_data_array[devid].mcxqueue = mcxqueue[devid];
				kernel_thread_data_array[devid].gparam = gparam[devid];
				kernel_thread_data_array[devid].param = param;
				kernel_thread_data_array[devid].mcxkernel = mcxkernel[devid];
				kernel_thread_data_array[devid].mcgrid = &mcgrid[0];
				kernel_thread_data_array[devid].mcblock = &mcblock[0];
				kernel_thread_data_array[devid].gdetected = gdetected[devid];
				kernel_thread_data_array[devid].detected = detected;



				printf("Creating thread for device %d\n", devid);
				rc = pthread_create(&threads[devid], &attr, Kernel_Exe, (void *)&kernel_thread_data_array[devid]);
				if (rc) {
					printf("ERROR; return code from pthread_create() is %d\n", rc);
					exit(-1);
				}
			}

			pthread_attr_destroy(&attr);
			for(cl_uint devid=0;devid<workdev;devid++){
				rc = pthread_join(threads[devid], &pstatus);
				if (rc) {
					printf("ERROR; return code from pthread_join() is %d\n", rc);
					exit(-1);
				}
				printf("Main: completed join with thread %d having a status of %ld\n", devid,(long)pstatus);
			}

			for(cl_uint devid=0;devid<workdev;devid++){
				detected += kernel_thread_data_array[devid].detected;
			}

			printf("detected: %d\n", detected);
			tic1=GetTimeMillis();
			toc+=tic1-tic0;
			fprintf(cfg->flog,"tic1-tic_pthread:  \t%d ms\n", tic1-tic_pthread);
			fprintf(cfg->flog,"toc:  \t%d ms\n", toc);
			fprintf(cfg->flog,"kernel complete:  \t%d ms\nretrieving flux ... \t",tic1-tic);

			// end pthread
/*
			clWaitForEvents(workdev,waittoread);
			tic1=GetTimeMillis();
			toc+=tic1-tic0;
			fprintf(cfg->flog,"kernel complete:  \t%d ms\nretrieving flux ... \t",tic1-tic);
*/

			for(cl_uint devid=0;devid<workdev;devid++){
				if(cfg->issavedet){
					OCL_ASSERT((clEnqueueReadBuffer(mcxqueue[devid],gdetphoton[devid],CL_TRUE,0,sizeof(float)*cfg->maxdetphoton*(cfg->medianum+1),
									Pdet, 0, NULL, NULL)));
					if(detected>cfg->maxdetphoton){
						fprintf(cfg->flog,"WARNING: the detected photon (%d) \
								is more than what your have specified (%d), please use the -H option to specify a greater number\t"
								,detected,cfg->maxdetphoton);
					}else{
						fprintf(cfg->flog,"detected %d photons, total: %d\t",detected,cfg->detectedcount+detected);
					}
					cfg->his.detected+=detected;
					detected=MIN(detected,cfg->maxdetphoton);
					if(cfg->exportdetected){
						cfg->exportdetected=(float*)realloc(cfg->exportdetected,(cfg->detectedcount+detected)*detreclen*sizeof(float));
						memcpy(cfg->exportdetected+cfg->detectedcount*(detreclen),Pdet,detected*(detreclen)*sizeof(float));
						cfg->detectedcount+=detected;
					}
				}
				//handling the 2pt distributions
				if(cfg->issave2pt){
					OCL_ASSERT((clEnqueueReadBuffer(mcxqueue[devid],gfield[devid],CL_TRUE,0,sizeof(cl_float)*dimxyz*cfg->maxgate,
									field, 0, NULL, NULL)));
					fprintf(cfg->flog,"transfer complete:\t%d ms\n",GetTimeMillis()-tic);  fflush(cfg->flog);

					if(cfg->respin>1){
						for(i=0;i<fieldlen;i++)  //accumulate field, can be done in the GPU
							field[fieldlen+i]+=field[i];
					}
					if(iter+1==cfg->respin){ 
						if(cfg->respin>1)  //copy the accumulated fields back
							memcpy(field,field+fieldlen,sizeof(cl_float)*fieldlen);
					}
					if(cfg->isnormalized){

						OCL_ASSERT((clEnqueueReadBuffer(mcxqueue[devid],genergy[devid],CL_TRUE,0,sizeof(cl_float)*(cfg->nthread<<1),
										energy, 0, NULL, NULL)));
						for(i=0;i<cfg->nthread;i++){
							cfg->energyesc+=energy[(i<<1)];
							cfg->energytot+=energy[(i<<1)+1];
							//eabsorp+=Plen[i].z;  // the accumulative absorpted energy near the source
						}
					}
					if(cfg->exportfield){
						for(i=0;i<fieldlen;i++)
							cfg->exportfield[i]+=field[i];
					}
				}
				//initialize the next simulation
				if(twindow1<cfg->tend && iter<cfg->respin){
					memset(field,0,sizeof(cl_float)*dimxyz*cfg->maxgate);
					OCL_ASSERT((clEnqueueWriteBuffer(mcxqueue[devid],gfield[devid],CL_TRUE,0,sizeof(cl_float)*dimxyz*cfg->maxgate,
									field, 0, NULL, NULL)));
					OCL_ASSERT((clSetKernelArg(mcxkernel[devid], 3, sizeof(cl_mem), (void*)(gfield+devid))));
				}
				if(cfg->respin>1 && RAND_SEED_LEN>1){
					for (i=0; i<cfg->nthread*RAND_SEED_LEN; i++)
						Pseed[i]=rand();
					OCL_ASSERT((clEnqueueWriteBuffer(mcxqueue[devid],gseed[devid],CL_TRUE,0,sizeof(cl_uint)*cfg->nthread*RAND_SEED_LEN,
									Pseed, 0, NULL, NULL)));
					OCL_ASSERT((clSetKernelArg(mcxkernel[devid], 5, sizeof(cl_mem), (void*)(gseed+devid))));
				}
				OCL_ASSERT((clFinish(mcxqueue[devid])));
			}// loop over work devices
		}// iteration
		if(twindow1<cfg->tend){
			cl_float *tmpenergy=(cl_float*)calloc(sizeof(cl_float),cfg->nthread*3);
			OCL_ASSERT((clEnqueueWriteBuffer(mcxqueue[devid],genergy[devid],CL_TRUE,0,sizeof(cl_float)*(cfg->nthread<<1),
							tmpenergy, 0, NULL, NULL)));
			OCL_ASSERT((clSetKernelArg(mcxkernel[devid], 4, sizeof(cl_mem), (void*)(genergy+devid))));	
			free(tmpenergy);
		}
	}// time gates

	if(cfg->isnormalized){
		float scale=0.f;
		fprintf(cfg->flog,"normalizing raw data ...\t");

		if(cfg->outputtype==otFlux || cfg->outputtype==otFluence){
			scale=1.f/(cfg->energytot*Vvox*cfg->tstep);
			if(cfg->unitinmm!=1.f)
				scale*=cfg->unitinmm; /* Vvox (in mm^3 already) * (Tstep) * (Eabsorp/U) */

			if(cfg->outputtype==otFluence)
				scale*=cfg->tstep;
		}else if(cfg->outputtype==otEnergy || cfg->outputtype==otJacobian)
			scale=1.f/cfg->energytot;

		fprintf(cfg->flog,"normalization factor alpha=%f\n",scale);  fflush(cfg->flog);
		mcx_normalize(cfg->exportfield,scale,fieldlen);
	}
	if(cfg->issave2pt && cfg->parentid==mpStandalone){
		fprintf(cfg->flog,"saving data to file ... %d %d\t",fieldlen,cfg->maxgate);
		mcx_savedata(cfg->exportfield,fieldlen,0,"mc2",cfg);
		fprintf(cfg->flog,"saving data complete : %d ms\n\n",GetTimeMillis()-tic);
		fflush(cfg->flog);
	}
	if(cfg->issavedet && cfg->parentid==mpStandalone && cfg->exportdetected){
		cfg->his.unitinmm=cfg->unitinmm;
		cfg->his.savedphoton=cfg->detectedcount;
		cfg->his.detected=cfg->detectedcount;
		mcx_savedetphoton(cfg->exportdetected,cfg->seeddata,cfg->detectedcount,0,cfg);
	}

	// total energy here equals total simulated photons+unfinished photons for all threads
	fprintf(cfg->flog,"simulated %d photons (%d) with %d CUs with %d threads (repeat x%d)\nMCX simulation speed: %.2f photon/ms\n",
			cfg->nphoton,cfg->nphoton,workdev,cfg->nthread, cfg->respin,(double)cfg->nphoton/toc); fflush(cfg->flog);
	fprintf(cfg->flog,"total simulated energy: %.2f\tabsorbed: %5.5f%%\n(loss due to initial specular reflection is excluded in the total)\n",
			cfg->energytot,(cfg->energytot-cfg->energyesc)/cfg->energytot*100.f);fflush(cfg->flog);
	fflush(cfg->flog);

	//clReleaseMemObject(gmedia);
	//clReleaseMemObject(gproperty);
	//clReleaseMemObject(gparam);

	for(i=0;i<workdev;i++){
		clReleaseMemObject(gfield[i]);
		clReleaseMemObject(gseed[i]);
		clReleaseMemObject(genergy[i]);
		clReleaseMemObject(gstopsign[i]);
		clReleaseMemObject(gdetected[i]);
		clReleaseMemObject(gdetpos[i]);

		clReleaseMemObject(gmedia[i]);
		clReleaseMemObject(gproperty[i]);
		clReleaseMemObject(gparam[i]);

	}

	free(gfield);
	free(gseed);
	free(genergy);
	free(gstopsign);
	free(gdetected);
	free(gdetpos);

	free(waittoread);

	for(cl_uint devid=0;devid<workdev;devid++)
	{
		clReleaseKernel(mcxkernel[devid]);
		clReleaseCommandQueue(mcxqueue[devid]);
	}

	//free(mcxkernel);
	free(mcxqueue);

	//clReleaseProgram(mcxprogram);
	//clReleaseContext(mcxcontext);

	for(cl_uint devid=0;devid<workdev;devid++)
	{
		clReleaseProgram(mcxprogram[devid]);
		clReleaseContext(mcxcontext[devid]);
	}


#ifndef USE_OS_TIMER
	clReleaseEvent(kernelevent);
#endif
	free(Pseed);
	free(energy);
	free(field);
}

////////////////////////////////////////////////////////////////////////////////
//
//  Monte Carlo eXtreme (MCX)  - GPU accelerated Monte Carlo 3D photon migration
//      -- OpenCL edition
//  Author: Qianqian Fang <fangq at nmr.mgh.harvard.edu>
//
//  Reference (Fang2009):
//        Qianqian Fang and David A. Boas, "Monte Carlo Simulation of Photon 
//        Migration in 3D Turbid Media Accelerated by Graphics Processing 
//        Units," Optics Express, vol. 17, issue 22, pp. 20178-20190 (2009)
//
//  mcx_core.cl: OpenCL kernels
//
//  Unpublished work, see LICENSE.txt for details
//
////////////////////////////////////////////////////////////////////////////////

#ifdef MCX_SAVE_DETECTORS
  #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#endif

#ifdef MCX_GPU_DEBUG
  #define GPUDEBUG(x)        printf x             // enable debugging in CPU mode
  #pragma OPENCL EXTENSION cl_amd_printf : enable
#else
  #define GPUDEBUG(x)
#endif

//#pragma OPENCL EXTENSION cl_amd_printf : enable

#define R_PI               0.318309886183791f
#define RAND_MAX           4294967295

#define ONE_PI             3.1415926535897932f     //pi
#define TWO_PI             6.28318530717959f       //2*pi

#define C0                 299792458000.f          //speed of light in mm/s
#define R_C0               3.335640951981520e-12f  //1/C0 in s/mm

#define EPS                FLT_EPSILON             //round-off limit
#define VERY_BIG           (1.f/FLT_EPSILON)       //a big number
#define JUST_ABOVE_ONE     1.0001f                 //test for boundary
#define SAME_VOXEL         -9999.f                 //scatter within a voxel
#define NO_LAUNCH          9999                    //when fail to launch, for debug
#define MAX_PROP           256                     //maximum property number

#define DET_MASK           0x80
#define MED_MASK           0x7F
#define NULL               0

typedef struct KernelParams {
  float4 ps,c0;
  float4 maxidx;
  uint4  dimlen,cp0,cp1;
  uint2  cachebox;
  float  minstep;
  float  twin0,twin1,tmax;
  float  oneoverc0;
  unsigned int isrowmajor,save2pt,doreflect,dorefint,savedet;
  float  Rtstep;
  float  minenergy;
  float  skipradius2;
  float  minaccumtime;
  unsigned int maxdetphoton;
  unsigned int maxmedia;
  unsigned int detnum;
  unsigned int idx1dorig;
  unsigned int mediaidorig;
} MCXParam __attribute__ ((aligned (32)));


#ifndef USE_XORSHIFT128P_RAND

#define RAND_BUF_LEN       5        //register arrays
#define RAND_SEED_LEN      5        //32bit seed length (32*5=160bits)
#define INIT_LOGISTIC      100

typedef float RandType;

#define FUN(x)               (4.f*(x)*(1.f-(x)))
#define NU                   1e-7f
#define NU2                  (1.f-2.f*NU)
#define MIN_INVERSE_LIMIT    1e-7f
#define logistic_uniform(v)  (acos(1.f-2.f*(v))*R_PI)
#define R_MAX_C_RAND         (1.f/RAND_MAX)
#define LOG_MT_MAX           22.1807097779182f

#define RING_FUN(x,y,z)      (NU2*(x)+NU*((y)+(z)))

void logistic_step(__private RandType *t, __private RandType *tnew, int len_1){
    t[0]=FUN(t[0]);
    t[1]=FUN(t[1]);
    t[2]=FUN(t[2]);
    t[3]=FUN(t[3]);
    t[4]=FUN(t[4]);
    tnew[4]=RING_FUN(t[0],t[4],t[1]);   /* shuffle the results by separation of 2*/
    tnew[0]=RING_FUN(t[1],t[0],t[2]);
    tnew[1]=RING_FUN(t[2],t[1],t[3]);
    tnew[2]=RING_FUN(t[3],t[2],t[4]);
    tnew[3]=RING_FUN(t[4],t[3],t[0]);
}
// generate random number for the next zenith angle
void rand_need_more(__private RandType t[RAND_BUF_LEN]){
    RandType tnew[RAND_BUF_LEN]={0.f};
    logistic_step(t,tnew,RAND_BUF_LEN-1);
    logistic_step(tnew,t,RAND_BUF_LEN-1);
}

void logistic_init(__private RandType *t,__global uint seed[],uint idx){
     int i;
     for(i=0;i<RAND_BUF_LEN;i++)
           t[i]=(RandType)seed[idx*RAND_BUF_LEN+i]*R_MAX_C_RAND;

     for(i=0;i<INIT_LOGISTIC;i++)  /*initial randomization*/
           rand_need_more(t);
}
// transform into [0,1] random number
RandType rand_uniform01(__private RandType t[RAND_BUF_LEN]){
    rand_need_more(t);
    return logistic_uniform(t[0]);
}
void gpu_rng_init(__private RandType t[RAND_BUF_LEN],__global uint *n_seed,int idx){
    logistic_init(t,n_seed,idx);
}

#else

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define RAND_BUF_LEN       2        //register arrays
#define RAND_SEED_LEN      4        //48 bit packed with 64bit length
#define LOG_MT_MAX         22.1807097779182f
#define IEEE754_DOUBLE_BIAS     0x3FF0000000000000ul /* Added to exponent.  */

typedef ulong  RandType;

float xorshift128p_nextf (__private RandType t[RAND_BUF_LEN]){
   union {
        double d;
        ulong  i;
   } s1;
   const ulong s0 = t[1];
   s1.i = t[0];
   t[0] = s0;
   s1.i ^= s1.i << 23; // a
   t[1] = s1.i ^ s0 ^ (s1.i >> 18) ^ (s0 >> 5); // b, c
   s1.i = t[1] + s0;
   s1.i = (s1.i >> 12) | IEEE754_DOUBLE_BIAS;

   return (float)s1.d - 1.0f;
}

void copystate(__private RandType t[RAND_BUF_LEN], __private RandType tnew[RAND_BUF_LEN]){
    tnew[0]=t[0];
    tnew[1]=t[1];
}

// generate random number for the next zenith angle
void rand_need_more(__private RandType t[RAND_BUF_LEN]){
}

float rand_uniform01(__private RandType t[RAND_BUF_LEN]){
    return xorshift128p_nextf(t);
}

void xorshift128p_seed (__global uint seed[4],RandType t[RAND_BUF_LEN])
{
    t[0] = (ulong)seed[0] << 32 | seed[1] ;
    t[1] = (ulong)seed[2] << 32 | seed[3];
}

void gpu_rng_init(__private RandType t[RAND_BUF_LEN], __global uint *n_seed, int idx){
    xorshift128p_seed((n_seed+idx*RAND_SEED_LEN),t);
}
void gpu_rng_reseed(__private RandType t[RAND_BUF_LEN],__global uint cpuseed[],uint idx,float reseed){
}

#endif

float rand_next_scatlen(__private RandType t[RAND_BUF_LEN]){
    return -log(rand_uniform01(t)+EPS);
}

#define rand_next_aangle(t)  rand_uniform01(t)
#define rand_next_zangle(t)  rand_uniform01(t)
#define rand_next_reflect(t) rand_uniform01(t)
#define rand_do_roulette(t)  rand_uniform01(t) 


#ifdef USE_ATOMIC
// OpenCL float atomicadd hack:
// http://suhorukov.blogspot.co.uk/2011/12/opencl-11-atomic-operations-on-floating.html

inline void atomicadd(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}
#endif

void clearpath(__local float *p, __constant MCXParam gcfg[]){
      uint i;
      for(i=0;i<gcfg->maxmedia;i++)
      	   p[i]=0.f;
}

#ifdef MCX_SAVE_DETECTORS
uint finddetector(float4 p0[],__constant float4 gdetpos[],__constant MCXParam gcfg[]){
      uint i;
      for(i=0;i<gcfg->detnum;i++){
      	if((gdetpos[i].x-p0[0].x)*(gdetpos[i].x-p0[0].x)+
	   (gdetpos[i].y-p0[0].y)*(gdetpos[i].y-p0[0].y)+
	   (gdetpos[i].z-p0[0].z)*(gdetpos[i].z-p0[0].z) < gdetpos[i].w){
	        return i+1;
	   }
      }
      return 0;
}

void savedetphoton(__global float n_det[],__global uint *detectedphoton,float nscat,
                   __local float *ppath,float4 p0[],__constant float4 gdetpos[],__constant MCXParam gcfg[]){
      uint detid;
      detid=finddetector(p0,gdetpos,gcfg);
      if(detid){
	 uint baseaddr=atomic_inc(detectedphoton);
	 if(baseaddr<gcfg->maxdetphoton){
	    uint i;
	    baseaddr*=gcfg->maxmedia+2;
	    n_det[baseaddr++]=detid;
	    n_det[baseaddr++]=nscat;
	    for(i=0;i<gcfg->maxmedia;i++){
		n_det[baseaddr+i]=ppath[i]; // save partial pathlength to the memory
	    }
	 }
      }
}
#endif

float mcx_nextafterf(float a, int dir){
      union{
          float f;
	  uint  i;
      } num;
      num.f=a+1000.f;
      num.i+=dir ^ (num.i & 0x80000000U);
      return num.f-1000.f;
}

float hitgrid(float4 p0[], float4 v[], float4 htime[], int *id){
      float dist;

      //time-of-flight to hit the wall in each direction

      htime[0]=fabs(floor(p0[0])+convert_float4(isgreater(v[0],((float4)(0.f))))-p0[0]);
      htime[0]=fabs(native_divide(htime[0]+(float4)EPS,v[0]));

      //get the direction with the smallest time-of-flight
      dist=fmin(fmin(htime[0].x,htime[0].y),htime[0].z);
      (*id)=(dist==htime[0].x?0:(dist==htime[0].y?1:2));

      htime[0]=p0[0]+(float4)(dist)*v[0];

      (*id==0) ?
          (htime[0].x=mcx_nextafterf(convert_int_rte(htime[0].x), (v[0].x > 0.f)-(v[0].x < 0.f))) :
	  ((*id==1) ? 
	  	(htime[0].y=mcx_nextafterf(convert_int_rte(htime[0].y), (v[0].y > 0.f)-(v[0].y < 0.f))) :
		(htime[0].z=mcx_nextafterf(convert_int_rte(htime[0].z), (v[0].z > 0.f)-(v[0].z < 0.f))) );
      return dist;
}


void rotatevector(float4 v[], float stheta, float ctheta, float sphi, float cphi){
      if( v[0].z>-1.f+EPS && v[0].z<1.f-EPS ) {
   	  float tmp0=1.f-v[0].z*v[0].z;
   	  float tmp1=stheta*rsqrt(tmp0);
   	  *((float4*)v)=(float4)(
   	       tmp1*(v[0].x*v[0].z*cphi - v[0].y*sphi) + v[0].x*ctheta,
   	       tmp1*(v[0].y*v[0].z*cphi + v[0].x*sphi) + v[0].y*ctheta,
   	      -tmp1*tmp0*cphi                          + v[0].z*ctheta,
   	       v[0].w
   	  );
      }else{
   	  v[0]=(float4)(stheta*cphi,stheta*sphi,(v[0].z>0.f)?ctheta:-ctheta,v[0].w);
      }
      GPUDEBUG(((__constant char*)"new dir: %10.5e %10.5e %10.5e\n",v[0].x,v[0].y,v[0].z));
}

int launchnewphoton(float4 p[],float4 v[],float4 f[],float4 prop[],uint *idx1d,
           uint *mediaid,float *w0,uchar isdet, __local float ppath[],float *energyloss,float *energylaunched,
	   __global float n_det[],__global uint *dpnum, __constant float4 gproperty[],
	   __constant float4 gdetpos[],__constant MCXParam gcfg[],int threadid, int threadphoton, int oddphotons){
      
      if(p[0].w>=0.f){
          *energyloss+=p[0].w;  // sum all the remaining energy
#ifdef MCX_SAVE_DETECTORS
          // let's handle detectors here
          if(gcfg->savedet){
             if(*mediaid==0 && isdet){
	          savedetphoton(n_det,dpnum,v[0].w,ppath,p,gdetpos,gcfg);
	     }
	     clearpath(ppath,gcfg);
          }
#endif
      }

      if(f[0].w>=(threadphoton+(threadid<oddphotons)))
         return 1; // all photons complete 
      p[0]=gcfg->ps;
      v[0]=gcfg->c0;
      f[0]=(float4)(0.f,0.f,gcfg->minaccumtime,f[0].w+1);
      *idx1d=gcfg->idx1dorig;
      *mediaid=gcfg->mediaidorig;
      prop[0]=gproperty[*mediaid & MED_MASK]; //always use mediaid to read gproperty[]
      *energylaunched+=p[0].w;
      *w0=p[0].w;
      return 0;
}

/*
   this is the core Monte Carlo simulation kernel, please see Fig. 1 in Fang2009
*/
__kernel void mcx_main_loop(const int nphoton, const int ophoton,__global const uchar media[],
     __global float field[], __global float genergy[], __global uint n_seed[],
     __global float n_det[],__constant float4 gproperty[],
     __constant float4 gdetpos[], __global uint stopsign[1],__global uint detectedphoton[1],
     __local float *sharedmem, __constant MCXParam gcfg[]){

     int idx= get_global_id(0);

	 if(idx == 0) {

		 float a=10.0f,ap,an;
		 ap=mcx_nextafterf(a,a+1.f);
		 an=mcx_nextafterf(a,a-1.f);
		 printf("     float       hex        float+       next+       float-       next-\n");
		 printf("%12.5f %08X %15.8e %08X %15.8e %08X - math lib\n", a, *(unsigned int*)&a, ap, *(unsigned int*)&ap, an, *(unsigned int*)&an);
	 
		 *((unsigned int *)&a)=0x3F800000;
		 ap=mcx_nextafterf(a,a+1.f);
		 an=mcx_nextafterf(a,a-1.f);
		 printf("%12.5f %08X %15.8e %08X %15.8e %08X - math lib\n", a, *(unsigned int*)&a, ap, *(unsigned int*)&ap, an, *(unsigned int*)&an);
		 a=0.f;
		 ap=mcx_nextafterf(a,a+1.f);
		 an=mcx_nextafterf(a,a-1.f);
		 printf("%12.5f %08X %15.8e %08X %15.8e %08X - math lib\n", a, *(unsigned int*)&a, ap, *(unsigned int*)&ap, an, *(unsigned int*)&an);
		 a=-10.f;
		 ap=mcx_nextafterf(a,a+1.f);
		 an=mcx_nextafterf(a,a-1.f);
		 printf("%12.5f %08X %15.8e %08X %15.8e %08X - math lib\n", a, *(unsigned int*)&a, ap, *(unsigned int*)&ap, an, *(unsigned int*)&an);
		 a=-10.f;
		 ap=mcx_nextafterf(a+1000.f,a+1001.f)-1000.f;
		 an=mcx_nextafterf(a+1000.f,a-1001.f)-1000.f;
		 printf("%12.5f %08X %15.8e %08X %15.8e %08X - math lib+offset\n", a, *(unsigned int*)&a, ap, *(unsigned int*)&ap, an, *(unsigned int*)&an);
		 a=10.f;
		 ap=mcx_nextafterf(a+1000.f,a+1001.f)-1000.f;
		 an=mcx_nextafterf(a+1000.f,a-1001.f)-1000.f;
		 printf("%12.5f %08X %15.8e %08X %15.8e %08X - math lib+offset\n", a, *(unsigned int*)&a, ap, *(unsigned int*)&ap, an, *(unsigned int*)&an);
		 a=-10.f;
		 ap=mcx_nextafterf(a,1);
		 an=mcx_nextafterf(a,-1);
		 printf("%12.5f %08X %15.8e %08X %15.8e %08X - my nextafter\n", a, *(unsigned int*)&a, ap, *(unsigned int*)&ap, an, *(unsigned int*)&an);
		 a=0.f;
		 ap=mcx_nextafterf(a,1);
		 an=mcx_nextafterf(a,-1);
		 printf("%12.5f %08X %15.8e %08X %15.8e %08X - my nextafter\n", a, *(unsigned int*)&a, ap, *(unsigned int*)&ap, an, *(unsigned int*)&an);
		 a=10.f;
		 ap=mcx_nextafterf(a,1);
		 an=mcx_nextafterf(a,-1);
		 printf("%12.5f %08X %15.8e %08X %15.8e %08X - my nextafter\n", a, *(unsigned int*)&a, ap, *(unsigned int*)&ap, an, *(unsigned int*)&an);


		 // built-in

		 printf("\nopencl nextafter\n");

		 a=10.0f;
		 ap=nextafter(a,a+1.f);
		 an=nextafter(a,a-1.f);
		 printf("     float       hex        float+       next+       float-       next-\n");
		 printf("\n\n%12.5f %08X %15.8e %08X %15.8e %08X - math lib\n", a, *(unsigned int*)&a, ap, *(unsigned int*)&ap, an, *(unsigned int*)&an);
	 
		 *((unsigned int *)&a)=0x3F800000;
		 ap=nextafter(a,a+1.f);
		 an=nextafter(a,a-1.f);
		 printf("%12.5f %08X %15.8e %08X %15.8e %08X - math lib\n", a, *(unsigned int*)&a, ap, *(unsigned int*)&ap, an, *(unsigned int*)&an);
		 a=0.f;
		 ap=nextafter(a,a+1.f);
		 an=nextafter(a,a-1.f);
		 printf("%12.5f %08X %15.8e %08X %15.8e %08X - math lib\n", a, *(unsigned int*)&a, ap, *(unsigned int*)&ap, an, *(unsigned int*)&an);
		 a=-10.f;
		 ap=nextafter(a,a+1.f);
		 an=nextafter(a,a-1.f);
		 printf("%12.5f %08X %15.8e %08X %15.8e %08X - math lib\n", a, *(unsigned int*)&a, ap, *(unsigned int*)&ap, an, *(unsigned int*)&an);
		 a=-10.f;
		 ap=nextafter(a+1000.f,a+1001.f)-1000.f;
		 an=nextafter(a+1000.f,a-1001.f)-1000.f;
		 printf("%12.5f %08X %15.8e %08X %15.8e %08X - math lib+offset\n", a, *(unsigned int*)&a, ap, *(unsigned int*)&ap, an, *(unsigned int*)&an);
		 a=10.f;
		 ap=nextafter(a+1000.f,a+1001.f)-1000.f;
		 an=nextafter(a+1000.f,a-1001.f)-1000.f;
		 printf("%12.5f %08X %15.8e %08X %15.8e %08X - math lib+offset\n", a, *(unsigned int*)&a, ap, *(unsigned int*)&ap, an, *(unsigned int*)&an);
		 a=-10.f;
		 ap=nextafter(a,1);
		 an=nextafter(a,-1);
		 printf("%12.5f %08X %15.8e %08X %15.8e %08X - my nextafter\n", a, *(unsigned int*)&a, ap, *(unsigned int*)&ap, an, *(unsigned int*)&an);
		 a=0.f;
		 ap=nextafter(a,1);
		 an=nextafter(a,-1);
		 printf("%12.5f %08X %15.8e %08X %15.8e %08X - my nextafter\n", a, *(unsigned int*)&a, ap, *(unsigned int*)&ap, an, *(unsigned int*)&an);
		 a=10.f;
		 ap=nextafter(a,1);
		 an=nextafter(a,-1);
		 printf("%12.5f %08X %15.8e %08X %15.8e %08X - my nextafter\n", a, *(unsigned int*)&a, ap, *(unsigned int*)&ap, an, *(unsigned int*)&an);
	 
	 }



}


#ifndef USE_CUDA
#define USE_CUDA
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include "render.h"
template<typename scalar>
static inline __device__ __host__ scalar numeric_max() {
	if((scalar)-1 > 0) return (scalar)-1;
	bool is_float = ((scalar)1.1 != (scalar)1);
	switch(sizeof(scalar)) {
	case 8:	if(is_float) {
			return (scalar)1.7976931348623157879e308;
		} else	return (scalar)9223372036854775807;
	case 4:	if(is_float) {
			return (scalar)3.40282346638528875558e38f;
		} else	return (scalar)2147483647;
	case 2:	return (scalar)32767;
	default:return (scalar)127;}
}
template<class T, uint64_t bufsize>
class vector_gpu {
public:
	__device__ vector_gpu(uint64_t n = 0):
	ptr(NULL), len(n), mutex(0) {
		if(n > 0) {
			n = allocate(len);
			ptr = (T*)malloc(sizeof(T) * n);
			if(ptr == NULL) len = 0;
		}
	}
	__device__ ~vector_gpu() {
		if(ptr != NULL) free(ptr);
	}
	__device__ uint64_t size() const {return len;}
	__device__ T &operator[](uint64_t i) const {
		return	ptr[i % len];
	}
	__device__ void clear() {
		while(ptr != NULL)
			if(atomicCAS(&mutex, 0, 1) == 0) {
				free(ptr); len = 0; ptr = NULL;
				atomicExch(&mutex, 0);
			}
	}
	__device__ bool push_back(T p) {
		bool inserted = true;
		bool blocked = true;
		while(blocked)
			if(atomicCAS(&mutex, 0, 1) == 0) {
				if(len % bufsize == 0) {
					T*tmp = (T*)malloc(sizeof(T) *(len+bufsize));
					if(inserted = (tmp != NULL)) {
						for(uint64_t i = 0; i < len; ++i)
							tmp[i] = ptr[i];
						free(ptr); ptr = tmp;
					}
				}
				if(inserted) ptr[len++] = p;
				atomicExch(&mutex, 0);
				blocked = false;
			}
		return	inserted;
	}
protected:
	inline __device__ uint64_t allocate(uint64_t n) {
		return ((n + bufsize - 1) % bufsize) * bufsize;
	}
	mutable T*ptr;
	uint64_t len;
	int mutex;
};
template<typename scalar,typename index>
__global__ void zbuffer_forward_kernel(index h,index w,index n,index f,
		const scalar *v, const index *tri,
		scalar *zbuf, vector_gpu<index,256> *ibuf,
		index *i, scalar *coeff, bool *vis, bool persp, scalar eps) {
	index	st = 0, ed = h*w;
	for(index i = st; i < ed; ++i)
		zbuf[i] = numeric_max<scalar>();
	zbuffer_forward<scalar,index,vector_gpu<index,256> >(
		h, w, n, f, v, tri, zbuf, ibuf, i, coeff, vis, persp, eps);
}
#include <iostream>
template<typename scalar,typename index>
bool zbuffer_forward_gpu(index h, index w, index n, index f,
                const scalar *v, const index *tri,
                index *ind, scalar *coeff, bool *vis,
                bool persp, scalar eps) {
	vector_gpu<index,256> *ibuf = NULL;
	scalar *zbuf = NULL;
	cudaMalloc((void**)&ibuf, sizeof(vector_gpu<index,256>) * h * w);
	if(ibuf == NULL) return false;
	cudaMemset(ibuf, 0, sizeof(vector_gpu<index,256>) * h * w);
	cudaMalloc((void**)&zbuf, sizeof(scalar) * h * w);
	if(zbuf == NULL) {cudaFree(ibuf); return false;}
	index	threads = 512;
	zbuffer_forward_kernel<scalar,index><<<1,threads>>>(h, w, n, f,
			v, tri, zbuf, ibuf, ind, coeff, vis, persp, eps);
	cudaError_t e = cudaGetLastError();
	if(e != cudaSuccess) std::cout << cudaGetErrorString(e) << std::endl;
	cudaFree(zbuf);
	cudaFree(ibuf);
	return	e == cudaSuccess;
}
#include <vector>
#define IMPLEMENT(scalar) \
template int64_t zbuffer_forward<scalar,int64_t,std::vector<int64_t> >( \
	int64_t,int64_t,int64_t,int64_t,const scalar*,const int64_t*,scalar*, \
	std::vector<int64_t>*,int64_t*,scalar*,bool*,bool,scalar); \
template bool zbuffer_forward_gpu<scalar,int64_t>(int64_t,int64_t,int64_t,int64_t, \
	const scalar*,const int64_t*,int64_t*,scalar*, bool*,bool,scalar);

IMPLEMENT(float)

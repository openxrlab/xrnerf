#ifndef _RENDER_H_
#define _RENDER_H_
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#ifdef USE_CUDA
static __device__ float atomicMin(float* address, float val) {
	int* address_as_i = (int*) address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_i, assumed,
			__float_as_int(fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

#endif
template<typename index>
inline __device__ bool split_for_loop(index &st, index &ed, index stride = 1) {
#ifdef __CUDA_ARCH__
	index num = gridDim.x * blockDim.x;
	num = (ed + num * stride - 1 - st) / (num * stride);
	st = st + (blockIdx.x*blockDim.x + threadIdx.x) * num * stride;
	ed = st + num * stride < ed ? st + num * stride : ed;
#endif
	return st < ed;
}
template<typename scalar,typename index>
__device__ __host__ unsigned char process_one_tri(const scalar v[9],
		index w, index h, index bbox[4],
		scalar Ainv[9], scalar eps, bool double_face = false) {
	scalar	umin = (scalar)w, vmin = (scalar)h, umax = 0, vmax = 0;
	if(v != NULL) for(unsigned char i = 0; i < 3; ++i)
		if(i == 0) {
			umax = umin = v[3*i];
			vmax = vmin = v[3*i+1];
		} else {
			if(umin > v[3*i])	umin = v[3*i];
			else if(umax < v[3*i])	umax = v[3*i];
			if(vmin > v[3*i+1])	vmin = v[3*i+1];
			else if(vmax <v[3*i+1])	vmax = v[3*i+1];
		}
	else	return false;
	if(bbox != NULL) {
		umin = floor(umin);
		umax =  ceil(umax);
		vmin = floor(vmin);
		vmax =  ceil(vmax);
		bbox[0] = (index)(umin <  0 ? 0  : umin);
		bbox[1] = (index)(umax >= w ? w-1: umax);
		bbox[2] = (index)(vmin <  0 ? 0  : vmin);
		bbox[3] = (index)(vmax >= h ? h-1: vmax);
		if(bbox[1] < bbox[0] || bbox[3] < bbox[2])
			return false;
	}
	if(Ainv == NULL) return false;
	unsigned char type = 0;
	Ainv[6] = v[3]*v[7]-v[6]*v[4];
	Ainv[7] = v[6]*v[1]-v[0]*v[7];
	Ainv[8] = v[0]*v[4]-v[3]*v[1];
	scalar	det = Ainv[6] + Ainv[7] + Ainv[8];
	if(!double_face && det > eps)
		return false;
	Ainv[0] = v[4]-v[7];
	Ainv[1] = v[7]-v[1];
	Ainv[2] = v[1]-v[4];
	Ainv[3] = v[6]-v[3];
	Ainv[4] = v[0]-v[6];
	Ainv[5] = v[3]-v[0];
	if(det <= eps && det >= -eps) {
		scalar l2[] = {
			Ainv[0]*Ainv[0]+Ainv[3]*Ainv[3],
			Ainv[1]*Ainv[1]+Ainv[4]*Ainv[4],
			Ainv[2]*Ainv[2]+Ainv[5]*Ainv[5]};
		unsigned char i = (l2[0] > l2[1] ? 0 : 1), j, k;
		i = (l2[i] > l2[2] ? i : 2);
		j = (i+1)%3;
		k = (j+1)%3;
		if(l2[i] > eps*eps) {
			type = (1<<j) + (1<<k);
			Ainv[j]  = -(Ainv[k]  = (v[3*k]  -v[3*j])  / l2[i]);
			Ainv[j+3]= -(Ainv[k+3]= (v[3*k+1]-v[3*j+1])/ l2[i]);
			Ainv[j+6]= (v[3*k]*(v[3*k]-v[3*j])+v[3*k+1]*(v[3*k+1]-v[3*j+1]))/l2[i];
			Ainv[k+6]= (v[3*j]*(v[3*j]-v[3*k])+v[3*j+1]*(v[3*j+1]-v[3*k+1]))/l2[i];
			scalar l = sqrt(l2[i]);
			Ainv[i]  = (v[3*j+1]- v[3*k+1]) / l;
			Ainv[i+3]= (v[3*k]  - v[3*j])   / l;
			Ainv[i+6]= (v[3*j]*v[3*k+1] - v[3*k]*v[3*j+1]) / l;
		} else {
			type = (1<<i);
			Ainv[0] = v[3*i];
			Ainv[1] = v[3*i+1];
		}
	} else {
		type = 7;
		for(unsigned char i = 0; i < 9; ++i)
			Ainv[i] /= det;
	}
	return type;
}
template<typename scalar>
__device__ __host__ bool normalize_coeff(scalar c[3], const scalar uv[2],
		const scalar Ainv[9], unsigned char t, scalar eps) {
	unsigned char i = 0, j = 1, k = 2;
	switch(t) {
	case 7:	c[0] = Ainv[0]*uv[0] + Ainv[3]*uv[1] + Ainv[6];
		c[1] = Ainv[1]*uv[0] + Ainv[4]*uv[1] + Ainv[7];
		c[2] = Ainv[2]*uv[0] + Ainv[5]*uv[1] + Ainv[8];
		return (c[0] >= -eps && c[1] >= -eps && c[2] >= -eps);
	case 3: case 5: case 6:
		i = (7-t)/2; j = (i+1)%3; k = (j+1)%3;
		c[0] = Ainv[0]*uv[0] + Ainv[3]*uv[1] + Ainv[6];
		c[1] = Ainv[1]*uv[0] + Ainv[4]*uv[1] + Ainv[7];
		c[2] = Ainv[2]*uv[0] + Ainv[5]*uv[1] + Ainv[8];
		if(c[i]*c[i] > eps*eps) return false;
		c[i] = 0;
		return (c[j] >= -eps && c[k] >= -eps);
	case 1: case 2: case 4:
		i = t/2; j = (i+1)%3; k = (j+1)%3;
		c[j] = (uv[0] - Ainv[0]);
		c[k] = (uv[1] - Ainv[1]);
		c[i] = (c[j]*c[j] + c[k]*c[k]);
		if(c[i] > eps*eps) return false;
		c[j] = c[k] = 0;
		c[i] = 1;
		return true;
	default:return false;}
}
template<typename scalar,typename index,class vector>
__device__ __host__ index zbuffer_forward(index h, index w, index n, index f,
		const scalar*v, const index *tri, scalar *zbuf, vector *ibuf,
		index *ind, scalar*coeff, bool*vis, bool persp, scalar eps) {
	index	st = 0, ed = n, count = 0;
#ifdef __CUDA_ARCH__
	split_for_loop<index>(st, ed);
#endif
	for(index i = st; i < ed; ++i) {
		scalar	x = v[3*i], y = v[3*i+1];
		if(persp) {
			if(v[3*i+2] <= eps) {
				vis[i] = false; continue;
			} else {
				x /= v[3*i+2];
				y /= v[3*i+2];
			}
		}
		x = floor(x); y = floor(y);
		if(x < 0 || y < 0 || x >= (scalar)w || y >= (scalar)h) {
			vis[i] = false; continue;
		} else {
			index j = (index)x + (index)y * w;
			vis[i] = true;
			ibuf[j].push_back(i);
		}
	}
	st = 0; ed = f;
#ifdef __CUDA_ARCH__
	__syncthreads();
	split_for_loop<index>(st, ed);
#endif
	scalar	Ainv[9], c[3], uv[2], z;
	index	bbox[4];
	unsigned char t = 0;
	for(index i = st; i < ed; ++i) {
		if((v[3*tri[3*i]  +2] <= eps
		||  v[3*tri[3*i+1]+2] <= eps
		||  v[3*tri[3*i+2]+2] <= eps) && persp)
			continue;
		scalar	v_[] = {
			v[3*tri[3*i]],  v[3*tri[3*i]+1],  v[3*tri[3*i]  +2],
			v[3*tri[3*i+1]],v[3*tri[3*i+1]+1],v[3*tri[3*i+1]+2],
			v[3*tri[3*i+2]],v[3*tri[3*i+2]+1],v[3*tri[3*i+2]+2]};
		if(persp) for(unsigned char j = 0; j < 3; ++j) {
			v_[3*j]  /= v_[3*j+2];
			v_[3*j+1]/= v_[3*j+2];
		}
		if((t = process_one_tri<scalar,index>(v_, w, h, bbox, Ainv, eps)))
		for(index y = bbox[2]; y <= bbox[3]; ++y)
			for(index x = bbox[0]; x <= bbox[1]; ++x) {
				++count;
				index j = x + y*w;
				uv[0] = (scalar)x;
				uv[1] = (scalar)y;
				if(normalize_coeff<scalar>(c, uv, Ainv, t, eps)) {
					if(persp) {
						c[0] /= v_[2]; c[1] /= v_[5]; c[2] /= v_[8];
						z = c[0] + c[1] + c[2];
						if(z <= eps) continue;
						c[0] /= z; c[1] /= z; c[2] /= z;
						z = 1./ z;
					} else	z = c[0]*v_[2] + c[2]*v_[5] + c[2]*v_[8];
#ifdef __CUDA_ARCH__
					if(atomicMin(zbuf + j, z) > z)
#else
					if(zbuf[j] > z)
#endif
					{	zbuf[j] = z;
						ind[j] = i;
						coeff[3*j]  = c[0];
						coeff[3*j+1]= c[1];
						coeff[3*j+2]= c[2];
					}
				}
				for(index k = 0; k < ibuf[j].size(); ++k) {
					if(ibuf[j][k] == tri[3*i]
					|| ibuf[j][k] == tri[3*i+1]
					|| ibuf[j][k] == tri[3*i+2])
						continue;
					uv[0] = v[3*ibuf[j][k]];
					uv[1] = v[3*ibuf[j][k]+1];
					if(persp) {
						uv[0] /= v[3*ibuf[j][k]+2];
						uv[1] /= v[3*ibuf[j][k]+2];
					}
					if(normalize_coeff<scalar>(c, uv, Ainv, t, eps)) {
						if(persp) {
							c[0] /= v_[2]; c[1] /= v_[5]; c[2] /= v_[8];
							z = c[0] + c[1] + c[2];
							if(z <= eps) continue;
							c[0] /= z; c[1] /= z; c[2] /= z;
							z = 1./ z;
						} else	z = c[0]*v_[2] + c[2]*v_[5] + c[2]*v_[8];
						if(z <= v[3*ibuf[j][k]+2])
							vis[ibuf[j][k]] = false;
					}
				}
			}
	}
	st = 0; ed = h*w;
#ifdef __CUDA_ARCH__
	__syncthreads();
	split_for_loop<index>(st, ed);
#endif
	for(index i = st; i < ed; ++i) ibuf[i].clear();
#ifdef __CUDA_ARCH__
	__syncthreads();
#endif
	return count;
}
#endif

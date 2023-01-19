#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>
#include "matrix.h"

#ifndef MAX
#define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif
template<typename scalar_t>
__device__ scalar_t search_nearest_proj(
		const scalar_t patch[9], scalar_t coeff[3], scalar_t precision = 1e-9) {
	scalar_t p[29];
/*	coeff[0] = coeff[1] = coeff[2] = 1./3;
	p[0] = coeff[0]*patch[0]+coeff[1]*patch[3]+coeff[2]*patch[6];
	p[1] = coeff[0]*patch[1]+coeff[1]*patch[4]+coeff[2]*patch[7];
	p[2] = coeff[0]*patch[2]+coeff[1]*patch[5]+coeff[2]*patch[8];
	return p[0]*p[0]+p[1]*p[1]+p[2]*p[2];
*/	unsigned char i = 0, j = 1, k = 2;
	for(i = 0; i < 3; ++i)
		for(j = i; j < 3; ++j) {
			p[20+j+3*i] = 0;
			for(k = 0; k < 3; ++k)
				p[20+j+3*i] += patch[k+i*3] * patch[k+j*3];
			p[20+i+3*j] = p[20+j+3*i];
		}
	p[0] = p[20]; p[1] = p[21]; p[2] = p[22]; p[3] = 1;
	p[4] = p[23]; p[5] = p[24]; p[6] = p[25]; p[7] = 1;
	p[8] = p[26]; p[9] = p[27]; p[10]= p[28]; p[11]= 1;
	p[12]= 1;     p[13]= 1;     p[14]= 1;     p[15]= 0;

	p[16]= 0;
	p[17]= 0;
	p[18]= 0;
	p[19]= 1;
	if(!solve4<scalar_t>(p, p+16, precision)) {
		p[0] = p[24]+p[28]-p[25]-p[27];
		p[1] = p[28]+p[20]-p[26]-p[22];
		p[2] = p[20]+p[24]-p[21]-p[23];
		i = (p[0] < p[1] ? 1 : 0);
		i = (p[i] < p[2] ? 2 : i);
		j = (i+1) % 3; k = 3-i-j;
		p[0] = p[20+4*j];  p[1] = p[20+3*j+k];p[2] = 1;
		p[3] = p[20+3*k+j];p[4] = p[20+4*k];  p[5] = 1;
		p[6] = 1;          p[7] = 1;          p[8] = 0;

		p[9] = 0;
		p[10]= 0;
		p[11]= 1;
		if(!solve3<scalar_t>(p, p+9, precision)) {
			coeff[i] = 0;
			coeff[j] =.5;
			coeff[k] =.5;
			return (p[20+4*j]+p[20+4*k]) / 2;
		} else if(p[9] < 0) {
			coeff[i] = 0;
			coeff[j] = 0;
			coeff[k] = 1;
			return p[20+4*k];
		} else if(p[10] < 0) {
			coeff[i] = 0;
			coeff[j] = 1;
			coeff[k] = 0;
			return p[20+4*j];
		} else {
			coeff[i] = 0;
			coeff[j] = p[9];
			coeff[k] = p[10];
			return	ABS(p[11]);
		}
	} else {
		i = (p[16]  > p[17] ? 1 : 0);
		i = (p[16+i]> p[18] ? 2 : i);
		if(p[16+i] < 0) {
			j = (i+1) % 3; k = 3-i-j;
			p[0] = p[20+4*j];  p[1] = p[20+3*j+k];p[2] = 1;
			p[3] = p[20+3*k+j];p[4] = p[20+4*k];  p[5] = 1;
			p[6] = 1;          p[7] = 1;          p[8] = 0;

			p[9] = 0;
			p[10]= 0;
			p[11]= 1;
			solve3<scalar_t>(p, p+9, precision);
			if(p[9] < 0) {
				coeff[i] = 0;
				coeff[j] = 0;
				coeff[k] = 1;
				return p[20+4*k];
			} else if(p[10] < 0) {
				coeff[i] = 0;
				coeff[j] = 1;
				coeff[k] = 0;
				return p[20+4*j];
			} else {
				coeff[i] = 0;
				coeff[j] = p[9];
				coeff[k] = p[10];
				return ABS(p[11]);
			}
		} else {
			coeff[0] = p[16];
			coeff[1] = p[17];
			coeff[2] = p[18];
			return ABS(p[19]);
		}
	}
}
template<typename scalar_t, typename index, unsigned char dim>
__global__ void insert_grid_surface_kernel(
        const scalar_t *points, const index *_surf, index n,
		scalar_t step, const scalar_t _min[dim], const index num[dim],
		index *surf_num, index *surf_idx = NULL) {

    // const scalar_t step = _step[0];
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(points == NULL || _surf == NULL || _min == NULL || num == NULL || surf_num == NULL
	|| dim <= 0 || step <= 0 || n <= 0 || id >= n)
        return;
    const index *surf = _surf + id * dim;

    index bbox[dim * 2], bbox_num = 1;
    for(unsigned char d = 0; d < dim; ++d) {
        scalar_t minmax[2] = {
            points[dim*surf[0] + d],
            points[dim*surf[0] + d]};
        for(unsigned char j = 1; j < dim; ++j)
            if(minmax[0] > points[dim*surf[j] + d])
                minmax[0] = points[dim*surf[j] + d];
            else if(minmax[1] < points[dim*surf[j] + d])
                minmax[1] = points[dim*surf[j] + d];
        scalar_t x = (minmax[0] - _min[d]) / step;
        bbox[d]     = x < 0 ? 0 : (x >= num[d] ? num[d] - 1 : (index)floor(x));
        x = (minmax[1] - _min[d]) / step;
        bbox[d+dim] =(x < 0 ? 0 : (x >= num[d] ? num[d] - 1 : (index)floor(x))) + 1;
        bbox_num *= (bbox[d+dim] - bbox[d]);
    }
    for(index j = 0; j < bbox_num; ++j) {
        index ind = 0, k = j;
        for(unsigned char d = 0; d < dim; ++d) {
            if(d > 0) ind *= num[d];
            ind += (bbox[d] + k % (bbox[d+dim] - bbox[d]));
            k /= (bbox[d+dim] - bbox[d] + 1e-8);
        }
        if(surf_idx == NULL)
            // ++surf_num[ind];
            atomicAdd(surf_num+ind, 1);
        else
            for(k = (ind == 0 ? 0 : surf_num[ind-1]); k < surf_num[ind]; ++k)
                if(atomicCAS(surf_idx+k, 0, id+1) == 0) {
                    // surf_idx[k] = i + 1;
                    // atomicExch(&surf_idx[k], i+1)
                    break;
                }
    }
}

template<typename scalar_t>
void print_tensor(at::Tensor tensor){
    int32_t size = tensor.size(0);
    if (size < 100)
        for (int i=0; i<size; i++){
            std::cout << tensor[i].item<scalar_t>() << " ";
        }
    else{
        // for (int i=0; i<3; i++)
        //     std::cout << tensor[i].item<scalar_t>() << " ";
        // std::cout << " ... ";
        // for (int i=-1; i>-4; i--)
        //     std::cout << tensor[i].item<scalar_t>() << " ";
        for (int i=0; i<size/16; i++){
            for (int j=0; j<16; j++)
                std::cout << tensor[i*16+j].item<scalar_t>() << " ";
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

at::Tensor insert_grid_surface_cuda(
    at::Tensor verts,
    at::Tensor faces,
    at::Tensor minmax,
    at::Tensor num,
    float step,
    at::Tensor tri_num
) {
    if(faces.sizes().size() != 2) faces = faces.reshape({-1,3});
	const int32_t num_faces = faces.size(0);

    const int threads = 512;
    const dim3 blocks (num_faces / threads + 1, 1, 1);

	tri_num.zero_();        // clear tri_num buffer
	AT_DISPATCH_FLOATING_TYPES(verts.type(), "insert_grid_surface_cuda", ([&] {
        insert_grid_surface_kernel<scalar_t, int32_t, 3><<<blocks, threads>>>(
            verts.data<scalar_t>(),
            faces.data<int32_t>(),
            num_faces,
            step,
            minmax.data<scalar_t>(),
            num.data<int32_t>(),
            tri_num.data<int32_t>(),
            NULL
        );
        }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in first insert_grid_surface_cuda: %s\n", cudaGetErrorString(err));

    tri_num.set_(at::_cast_Int(tri_num.cumsum(0)));     // cumsum determines the size of tri_idx buffer

    // make buffer
    const int32_t size = tri_num[-1].item<int32_t>();
    // tri_idx.resize_({size});
    // tri_idx.zero_();
	at::Tensor tri_idx = at::zeros({size}, tri_num.options());
    AT_DISPATCH_FLOATING_TYPES(verts.type(), "insert_grid_surface_cuda2", ([&] {
        insert_grid_surface_kernel<scalar_t, int32_t, 3><<<blocks, threads>>>(
            verts.data<scalar_t>(),
            faces.data<int32_t>(),
            num_faces,
            step,
            minmax.data<scalar_t>(),
            num.data<int32_t>(),
            tri_num.data<int32_t>(),
            tri_idx.data<int32_t>()
        );
        }));

    err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in second insert_grid_surface_cuda: %s\n", cudaGetErrorString(err));

	return tri_idx;

}

template<typename scalar_t, typename index, unsigned char dim>
__global__ void search_nearest_point_kenerel(
		const index *tri_num, const index *tri_idx, const index *size,
		const scalar_t *_min, scalar_t step,
		const scalar_t *points_base, const index *tri,
		const scalar_t *point_search_, const index points_num,
		scalar_t *coeff_ = NULL, scalar_t *proj_ = NULL,
		index *near_idx_ = NULL, scalar_t max_r2 = 0)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const scalar_t *point_search = point_search_ + 3 * id;
	scalar_t *coeff = coeff_ + 3 * id;
	scalar_t *proj = proj_ + 3 * id;
    index *near_idx = near_idx_ + id;

	if(points_base == NULL || tri == NULL || point_search_ == NULL
	|| tri_num == NULL || tri_idx == NULL || size == NULL || _min == NULL
	|| step <= 0 || id >= points_num)
        return;

	index x[dim*2+1], maxLinf = 0, n = 1, nearest = tri_num[size[dim]-1];
	for(unsigned char d = 0; d < dim; ++d) {
		scalar_t xf = (point_search[d] - _min[d]) / step;
		xf = (xf < 0 ? 0 :(xf >= size[d] ? size[d]-1 : floor(xf)));
		x[d] = (index)xf;
		x[dim] = d > 0 ? x[dim] * size[d] + x[d] : x[d];
		if(x[d] > size[d] - x[d])
			maxLinf = MAX(maxLinf, x[d]);
		else
			maxLinf = MAX(maxLinf, size[d]-x[d]);
	}
	scalar_t dist2 = 0, e = 0, dis2 = (max_r2 <= 0 ? -1 : max_r2);
	for(index Linf = 0; Linf < maxLinf; ++Linf) {
		n = 1;
		for(unsigned char d = 1; d < dim; ++d)
			n *= (2*Linf+1);
		for(index f = 0; f < (Linf == 0 ? 1 : 2*dim); ++f) {
			x[dim+1+f%dim] = f < dim ? -Linf : Linf;
			for(index k = 0; k < n; ++k) {
				index i, j = k;
				for(unsigned char d = 1; d < dim; ++d) {
					if(d+f >= 2*dim) {
						x[dim+1+(d+f)%dim] = j%(2*Linf-1) - Linf + 1;
						j = j / (2*Linf-1);
					} else if(d+f >= dim) {
						x[dim+1+(d+f)%dim] = j%(2*Linf) - Linf + 1;
						j = j / (2*Linf);
					} else {
						x[dim+1+(d+f)%dim] = j%(2*Linf+1) - Linf;
						j = j / (2*Linf+1);
					}
				}
				dist2 = 0;
				for(unsigned char d = 0; d < dim; ++d) {
					index y = x[d] + x[dim+1+d];
					if(y < 0 || y >= size[d]) {
						x[dim] = size[dim]; break;
					}
					if(x[dim+1+d] < 0) {
						e = point_search[d] - _min[d] - step*(y+1);
						dist2 += e * e;
					} else if(x[dim+1+d] > 0) {
						e =-point_search[d] + _min[d] + step*y;
						dist2 += e * e;
					}
					x[dim] = d > 0 ? x[dim] * size[d] + y : y;
				}
				if(x[dim] >= size[dim]) continue;
				if(dis2 >= 0 && dis2 < dist2) continue;
				// Find closest point and distance in a triangle face
				for(i = x[dim] == 0 ? 0 : tri_num[x[dim]-1]; i < tri_num[x[dim]]; ++i) {
					scalar_t patch[dim * dim];
					scalar_t _coeff[dim] = {0.33,0.33,0.33};
					for(unsigned char d = 0; d < dim; ++d){
						for(unsigned char d_= 0; d_< dim; ++d_){
							patch[d_+ d*dim] = points_base[d_+dim*
								tri[d+dim*tri_idx[i]-dim]] -
								point_search[d_];
						}
					}
					dist2 = search_nearest_proj<scalar_t>(patch, _coeff);
// printf("%d: %f %f %f\n", (int)threadIdx.x, _coeff[0], _coeff[1], _coeff[2]);
					if(dis2 < 0 || dist2 < dis2) {
						if(coeff != NULL) {
							coeff[0] = _coeff[0];
							coeff[1] = _coeff[1];
							coeff[2] = _coeff[2];
							proj[0] = point_search[0] +
								coeff[0]*patch[0] +
								coeff[1]*patch[3] +
								coeff[2]*patch[6];
							proj[1] = point_search[1] +
								coeff[0]*patch[1] +
								coeff[1]*patch[4] +
								coeff[2]*patch[7];
							proj[2] = point_search[2] +
								coeff[0]*patch[2] +
								coeff[1]*patch[5] +
								coeff[2]*patch[8];
						}
						nearest = tri_idx[i] - 1;
						dis2 = dist2;
					}
				}
			}
			if(f < dim-1)
				n = n / (2*Linf+1) * (2*Linf);
			else if(f >= dim)
				n = n / (2*Linf) * (2*Linf-1);
		}
		if(dis2 >= 0 && dis2 < Linf*Linf*step*step) break;
	}
	// return nearest;
    near_idx[0] = nearest;
}

void search_nearest_point_cuda (
    at::Tensor points,
    at::Tensor verts,
    at::Tensor faces,
    at::Tensor tri_num,
    at::Tensor tri_idx,
    at::Tensor num,
    at::Tensor minmax,
    float step,
	at::Tensor near_faces,
	at::Tensor near_pts,
	at::Tensor coeff
) {
    if(points.sizes().size() != 2) points = points.reshape({-1,3});
    int32_t points_num = points.size(0);

    const int threads = 512;
    const dim3 blocks (points_num / threads + 1, 1, 1);

    // make output
    // near_faces.resize_({points_num});
	// near_faces.zero_();
    // near_pts.resize_({points_num, 3});
    // near_pts.zero_();
    // coeff.resize_({points_num, 3});
    // coeff.zero_();

    AT_DISPATCH_FLOATING_TYPES(verts.type(), "search_nearest_point_cuda", ([&] {
        search_nearest_point_kenerel<scalar_t, int32_t, 3><<<blocks, threads>>>(
            tri_num.data<int32_t>(),
            tri_idx.data<int32_t>(),
            num.data<int32_t>(),
            minmax.data<scalar_t>(),
            step,
            verts.data<scalar_t>(),
            faces.data<int32_t>(),
            points.data<scalar_t>(),
            points_num,
			coeff.data<scalar_t>(),
			near_pts.data<scalar_t>(),
            near_faces.data<int32_t>()
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in search_nearest_point_cuda: %s\n", cudaGetErrorString(err));

}

template<typename scalar_t>
bool __device__ intersect_tri(
		const scalar_t* src, unsigned char dir,
		scalar_t* patch, unsigned char dim
) {
	if(dir > 2 * dim) return false;
	bool intersect = false;
	scalar_t patch_[6], det = 0;
	switch(dir % 2) {
	case 1:	for(unsigned char d = 0; d < dim; ++d)
			if(patch[dir/2+dim*d] > src[dir/2]) {
				intersect = true; break;
			}
		if(!intersect) return false;
		break;
	default:for(unsigned char d = 0; d < dim; ++d)
			if(patch[dir/2+dim*d] < src[dir/2]) {
				intersect = true; break;
			}
		if(!intersect) return false;
		break;}
	if(dim <= 1) {
		return true;
	} else if(dim > 2) {
		unsigned char r = 0;
		for(unsigned char d = 0; d < dim; ++d) {
			for(unsigned char i = 0; i < dim-1; ++i) {
				patch_[i] = src[(i+1+dir/2)%dim];
				for(unsigned char j = 1; j < dim; ++j)
					patch_[i+(dim-1)*j] =
						patch[(i+1+dir/2)%dim+dim*((j+d)%dim)];
			}
			r += intersect_tri<scalar_t>(patch_, 0, patch_+dim-1, dim-1);
		}
		if(r % 2 == 0) return false;
	}
	for(unsigned char i = 0; i < dim*dim; ++i)
		patch[i] -= src[i%dim];
	/* For 3-dimension, dir % 2 == 0, dir / 2 == 0, we have
		[Xa Xb Xc 1][  Ca  ]   [X]    Ca  >= 0
		[Ya Yb Yc 0][  Cb  ] = [Y],   Cb  >= 0
		[Za Zb Zc 0][  Cc  ]   [Z]    Cc  >= 0
		[ 1  1  1 0][lambda]   [1]  lambda>= 0
	solve	[Xa-X Xb-X Xc-X 1]-1[0]
		[Ya-Y Yb-Y Yc-Y 0]  [0]
		[Za-Z Zb-Z Zc-Z 0]  [0] >= 0
		[  1    1    1  0]  [1]
	For arbitrary case, (i = dir/2)
		[V   ei]-1[0] =    V^-1ei ( bigger than 0 if dir%2==0 else 1)
		[e^T  0]  [1]   e^TV^-1ei
	*/
	switch(dim) {
	case 2:	patch_[0] = (dir/2==0 ? patch[3]:-patch[2]);
		patch_[1] = (dir/2==0 ?-patch[1]: patch[0]);
		det = patch[0]*patch[3] - patch[1]*patch[2];
		break;
	case 3: patch_[0] = (dir/2==0 ?
			patch[4]*patch[8]-patch[5]*patch[7] : (dir/2==1 ?
			patch[5]*patch[6]-patch[3]*patch[8] :
			patch[3]*patch[7]-patch[4]*patch[6]));
		patch_[1] = (dir/2==0 ?
			patch[2]*patch[7]-patch[1]*patch[8] : (dir/2==1 ?
			patch[0]*patch[8]-patch[2]*patch[6] :
			patch[1]*patch[6]-patch[0]*patch[7]));
		patch_[2] = (dir/2==0 ?
			patch[1]*patch[5]-patch[2]*patch[4] : (dir/2==1 ?
			patch[2]*patch[3]-patch[0]*patch[5] :
			patch[0]*patch[4]-patch[1]*patch[3]));
		det =	patch_[0]*patch[dir/2] +
			patch_[1]*patch[dir/2+3] +
			patch_[2]*patch[dir/2+6];
		break;
	default:for(unsigned char d = 0; d < dim; ++d)
			patch_[d] = (d == dir/2) ? 1 : 0;
		// Gauss elimination
		for(unsigned char i = 0; i < dim; ++i) {
			unsigned char pivot = i;
			for(unsigned char j = i + 1; j < dim; ++j)
				if(ABS(patch[pivot+dim*i]) < ABS(patch[j+dim*i]))
					pivot = j;
			if(ABS(patch[pivot+dim*i]) <= 0) return false;
			for(unsigned char j = 0; j < dim; ++j)
			if(j != pivot) {
				scalar_t factor = patch[j+dim*i] / patch[pivot+dim*i];
				for(unsigned char k = i+1; k < dim; ++k)
					patch[j+dim*k] -= factor * patch[pivot+dim*k];
				patch_[j] -= factor * patch[pivot];
			}
			if(i != pivot) {
				for(unsigned char k = i; k < dim; ++k) {
					det = patch[i+dim*k];
					patch[i+dim*k] = patch[pivot+dim*k];
					patch[pivot+dim*k] = det;
				}
				det = patch_[i];
				patch_[i] = patch_[pivot];
				patch_[pivot] = det;
			}
		}
		det = 1; break;}
	if(det == 0) return false;
	intersect = (det > 0) ^ (dir % 2);
	for(unsigned char d = 0; d < dim; ++d)
		if(intersect ^ (patch_[d] < 0))
			return false;
	return true;
}

template<typename scalar_t, typename index, unsigned char dim>
void __global__ search_inside_mesh_kernel(const index *tri_num, const index *tri_idx, const index *size,
		const scalar_t *_min, scalar_t step,
		const scalar_t *points_base, const index *tri,
		const scalar_t *points_query, const index points_num,
		scalar_t *signs) {

	const int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(points_base == NULL || tri == NULL || points_query == NULL
		|| tri_num == NULL || tri_idx == NULL || size == NULL || _min == NULL
		|| step <= 0 || id >= points_num)
		return;

	const scalar_t *point = points_query + 3 * id;
	scalar_t *sign = signs + id;
	index	x[dim+1], to_end[2*dim];
	scalar_t	patch[dim*dim];
	unsigned char out_dim = 0;
	for(unsigned char d = 0; d < dim; ++d) {
		scalar_t xf = (point[d] - _min[d]) / step;
		if(xf < 0 || xf >= size[d]){
			// return false;
			sign[0] = -1;
			return;
		}
		x[d] = (index)xf;
		to_end[2*d]  = x[d];
		to_end[2*d+1]= size[d]-1-x[d];
		x[dim] = d > 0 ? x[dim] * size[d] + x[d] : x[d];
	}
	for(unsigned char d = 1; d < 2*dim; ++d)
		if(to_end[d] < to_end[out_dim])
			out_dim = d;
	// std::vector<index> visited(1, 0);
	// thrust::device_vector<index> visited(1, 0);
	index visited[16] = {};
	index visited_size = 1;
	for(index i = 0; i <= to_end[out_dim]; ++i) {
		for(index j =(x[dim]==0?0:tri_num[x[dim]-1]); j < tri_num[x[dim]]; ++j) {
			for(unsigned char d = 0; d < dim; ++d)
				for(unsigned char d_= 0; d_< dim; ++d_)
					patch[d_+ d*dim] = points_base[d_+dim*
						tri[d+dim*tri_idx[j]-dim]];
			if(intersect_tri<scalar_t>(point, out_dim, patch, dim)) {
				bool find = false;
				for(index t = 1; t < visited_size; ++t)
					if(visited[t] == tri_idx[j]-1) {
						find = true; break;
					}
				if(!find) {
					// visited.resize(visited.size()+1);
					// visited[visited.size()-1] = tri_idx[j]-1;
					if(visited_size < sizeof(visited)/sizeof(visited[0]))
						visited[visited_size++] = tri_idx[j]-1;
					else {	for(index i = 1; i+1 <
						sizeof(visited)/sizeof(visited[0]); ++i)
							visited[i] = visited[i+1];
						visited[sizeof(visited)/sizeof(visited[0])-1]
							= tri_idx[j]-1;
						visited_size++;
					}
				}
			}
		}
		if(out_dim % 2 == 1)
			++x[out_dim/2];
		else	--x[out_dim/2];
		for(unsigned char d = 0; d < dim; ++d)
			x[dim] = d > 0 ? x[dim] * size[d] + x[d] : x[d];
	}
	// return visited.size()-1;
	sign[0] = ((visited_size) % 2 == 0) ? 1 : -1;
}

void search_inside_mesh_cuda (
    at::Tensor points,
    at::Tensor verts,
    at::Tensor faces,
    at::Tensor tri_num,
    at::Tensor tri_idx,
    at::Tensor num,
    at::Tensor minmax,
    float step,
	at::Tensor signs
) {
    if(points.sizes().size() != 2) points = points.reshape({-1,3});
    int32_t points_num = points.size(0);

    const int threads = 512;
	const dim3 blocks (points_num / threads + 1, 1, 1);

    // make output
    // signs.resize_({points_num});
	// signs.zero_();

    AT_DISPATCH_FLOATING_TYPES(verts.type(), "search_inside_mesh_cuda", ([&] {
		search_inside_mesh_kernel<scalar_t, int32_t, 3><<<blocks, threads>>>(
            tri_num.data<int32_t>(),
            tri_idx.data<int32_t>(),
            num.data<int32_t>(),
            minmax.data<scalar_t>(),
            step,
            verts.data<scalar_t>(),
            faces.data<int32_t>(),
            points.data<scalar_t>(),
			points_num,
			signs.data<scalar_t>()
		);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in search_inside_mesh_cuda: %s\n", cudaGetErrorString(err));

}

template<typename scalar_t, typename index, unsigned char dim>
unsigned char __device__ ray_intersect_grid(
		const scalar_t start[dim], const scalar_t direction[dim],
		scalar_t step, const scalar_t min_[dim], const index num[dim + 1],
		index ind, bool first = false, scalar_t inter_point[dim] = NULL) {
	scalar_t _min[dim], _max[dim];
	if(ind < num[dim]) {
		for(unsigned char d = dim - 1; d > 0; ind /= num[d--])
			_max[d] = (_min[d] = min_[d] + step * (ind % num[d])) + step;
		_max[0] = (_min[0] = min_[0] + step * ind) + step;
	} else	for(unsigned char d = 0; d < dim; ++d)
			_max[d] = (_min[d] = min_[d]) + step * num[d];
	scalar_t min_dot = -1, point[dim];
	unsigned out_dim = 2 * dim;
	for(unsigned char d = 0; d < dim; ++d) {
		const scalar_t *inter;
		if(first) {
			if(start[d] < _min[d] && direction[d] > 0)
				inter = _min;
			else if(start[d] > _max[d] && direction[d] < 0)
				inter = _max;
			else if(start[d] > _min[d] && direction[d] < 0)
				inter = _min;
			else if(start[d] < _max[d] && direction[d] > 0)
				inter = _max;
			else
				continue;
		} else {
			if(direction[d] > 0)
				inter = _max;
			else if(direction[d] < 0)
				inter = _min;
			else
				continue;
		}
		scalar_t dot = (inter[d] - start[d]) / direction[d];
		if(dot < 0) continue;
		for(unsigned char d_= 0; d_< dim; ++d_)
			if(d_ != d) {
				point[d_] = start[d_] + direction[d_] * dot;
				if(point[d_] < _min[d_] || point[d_] > _max[d_]) {
					dot = min_dot; break;
				}
			} else
				point[d_] = inter[d_];
		if(dot >= 0 && (min_dot < 0 || dot < min_dot)) {
			min_dot = dot;
			out_dim = 2 * d + (inter == _max);
			if(inter_point != NULL)
				for(unsigned char d_= 0; d_< dim; ++d_)
					inter_point[d_] = point[d_];
		}
	}
	return out_dim;
}

template<typename scalar_t>
__device__ bool intersect_tri2(const scalar_t src[3], const scalar_t dir[3],
		const scalar_t va[3], const scalar_t vb[3], const scalar_t vc[3],
		scalar_t coeff[3] = NULL, bool both_direction = false,
		scalar_t precision = 1e-9) {
	scalar_t	A[] = {	va[0]-src[0], vb[0]-src[0], vc[0]-src[0], -dir[0],
			va[1]-src[1], vb[1]-src[1], vc[1]-src[1], -dir[1],
			va[2]-src[2], vb[2]-src[2], vc[2]-src[2], -dir[2],
			1, 1, 1, 0},
		A3inv[9],
		Ainv[4];
	A3inv[0] = A[5]*A[10]- A[6]*A[9];
	A3inv[1] = A[2]*A[9] - A[1]*A[10];
	A3inv[2] = A[1]*A[6] - A[2]*A[5];

	A3inv[3] = A[6]*A[8] - A[4]*A[10];
	A3inv[4] = A[0]*A[10]- A[2]*A[8];
	A3inv[5] = A[2]*A[4] - A[0]*A[6];

	A3inv[6] = A[4]*A[9] - A[5]*A[8];
	A3inv[7] = A[1]*A[8] - A[0]*A[9];
	A3inv[8] = A[0]*A[5] - A[1]*A[4];

	Ainv[0] =-A[3]*A3inv[0] - A[7]*A3inv[1] - A[11]*A3inv[2];
	Ainv[1] =-A[3]*A3inv[3] - A[7]*A3inv[4] - A[11]*A3inv[5];
	Ainv[2] =-A[3]*A3inv[6] - A[7]*A3inv[7] - A[11]*A3inv[8];
	Ainv[3] = A[0]*A3inv[0] + A[4]*A3inv[1] + A[8]*A3inv[2];
	scalar_t det = Ainv[0] + Ainv[1] + Ainv[2];
	if(det > precision || det < -precision) {
		if(coeff != NULL) {
			coeff[0] = Ainv[0] / det;
			coeff[1] = Ainv[1] / det;
			coeff[2] = Ainv[2] / det;
//			coeff[3] = Ainv[3] / det;
		}
		if(det < 0) {
			for(unsigned i = 0; i < 4; ++i)
				Ainv[i] = -Ainv[i];
			det = -det;
		}
		return	Ainv[0] >=-precision &&
			Ainv[1] >=-precision &&
			Ainv[2] >=-precision &&
			(both_direction || Ainv[3] >=-precision);
	} else {
		scalar_t	norm = A[3]*A[3] + A[7]*A[7] + A[11]*A[11],
			S[] = {
				A3inv[0] + A3inv[3] + A3inv[6],
				A3inv[1] + A3inv[4] + A3inv[7],
				A3inv[2] + A3inv[5] + A3inv[8]},
			area = S[0]*S[0] + S[1]*S[1] + S[2]*S[2];
		if(norm <= precision) {
		// direction degenerate to a point
			if(area > precision) {
				Ainv[0] = A3inv[0]*S[0]+A3inv[1]*S[1]+A3inv[2]*S[2];
				Ainv[1] = A3inv[3]*S[0]+A3inv[4]*S[1]+A3inv[5]*S[2];
				Ainv[2] = A3inv[6]*S[0]+A3inv[7]*S[1]+A3inv[8]*S[2];
				if(coeff != NULL) {
					coeff[0] = Ainv[0] / area;
					coeff[1] = Ainv[1] / area;
					coeff[2] = Ainv[2] / area;
//					coeff[3] = 0;
				}
				return	Ainv[0] >=-precision &&
					Ainv[1] >=-precision &&
					Ainv[2] >=-precision &&
					Ainv[3] >=-precision && Ainv[3] <= precision;
			} else {
				scalar_t	e[] = {	vc[0]-vb[0], vc[1]-vb[1], vc[2]-vb[2],
						va[0]-vc[0], va[1]-vc[1], va[2]-vc[2],
						vb[0]-va[0], vb[1]-va[1], vb[2]-va[2]},
					l[] = {	e[0]*e[0] + e[1]*e[1] + e[2]*e[2],
						e[3]*e[3] + e[4]*e[4] + e[5]*e[5],
						e[6]*e[6] + e[7]*e[7] + e[8]*e[8]};
				unsigned i = (l[0] < l[1] ? 1 : 0), j, k;
				i = (l[i] < l[2] ? 2 : i);
				j = (i+1) % 3;
				k = (i+2) % 3;
				if(l[i] > precision) {
				// triangle degenerate to a segment
					Ainv[i] = A3inv[3*i] * A3inv[3*i]  +
						A3inv[3*i+1] * A3inv[3*i+1]+
						A3inv[3*i+2] * A3inv[3*i+2];
					Ainv[j] = A[k]*e[3*i] + A[k+4]*e[3*i+1] + A[k+8]*e[3*i+2];
					Ainv[k] =-A[j]*e[3*i] - A[j+4]*e[3*i+1] - A[j+8]*e[3*i+2];
					if(coeff != NULL) {
						coeff[i] = 0;
						coeff[j] = Ainv[j] / l[i];
						coeff[k] = Ainv[k] / l[i];
//						coeff[3] = 0;
					}
					return	Ainv[i] <= precision &&
						Ainv[j] >=-precision &&
						Ainv[k] >=-precision &&
						Ainv[3] >=-precision && Ainv[3] <= precision;
				} else {
				// triangle degenerate to a point
					Ainv[i] = A[i]*A[i] + A[i+4]*A[i+4] + A[i+8]*A[i+8];
					if(coeff != NULL) {
						coeff[i] = 1;
						coeff[j] = 0;
						coeff[k] = 0;
//						coeff[3] = 0;
					}
					return	Ainv[i] <= precision &&
						Ainv[3] >=-precision && Ainv[3] <= precision;
				}
			}
		} else {
			if(area <= precision) {
				scalar_t	e[] = {	vc[0]-vb[0], vc[1]-vb[1], vc[2]-vb[2],
						va[0]-vc[0], va[1]-vc[1], va[2]-vc[2],
						vb[0]-va[0], vb[1]-va[1], vb[2]-va[2]},
					l[] = {	e[0]*e[0] + e[1]*e[1] + e[2]*e[2],
						e[3]*e[3] + e[4]*e[4] + e[5]*e[5],
						e[6]*e[6] + e[7]*e[7] + e[8]*e[8]};
				unsigned i = (l[0] < l[1] ? 1 : 0), j, k;
				i = (l[i] < l[2] ? 2 : i);
				j = (i+1) % 3;
				k = (i+2) % 3;
				if(l[i] <= precision) {
				// triangle degenerate to a point
					scalar_t	cross[] = {
							A[i+4]*A[11]-A[i+8]*A[7],
							A[i+8]*A[3] -A[i]  *A[11],
							A[i]  *A[7] -A[i+4]*A[3]};
					Ainv[i]=cross[0] * cross[0] +
						cross[1] * cross[1] +
						cross[2] * cross[2];
					Ainv[3]=-A[i]*A[3] - A[i+4]*A[7] - A[i+8]*A[11];
					if(coeff != NULL) {
						coeff[i] = 1;
						coeff[j] = 0;
						coeff[k] = 0;
//						coeff[3] = Ainv[3] / norm;
					}
					return	Ainv[i] <= precision &&
						(both_direction || Ainv[3] >=-precision);
				} else {
				// triangle degenerate to a segment
					scalar_t norm_ =
						A3inv[3*i]  * A3inv[3*i]  +
						A3inv[3*i+1]* A3inv[3*i+1]+
						A3inv[3*i+2]* A3inv[3*i+2];
					if(norm_ > precision) {
						scalar_t cross[] = {
							A[j+4]*A[11]-A[j+8]*A[7],
							A[j+8]*A[3] -A[j]  *A[11],
							A[j]  *A[7] -A[j+4]*A[3],
							A[k+4]*A[11]-A[k+8]*A[7],
							A[k+8]*A[3] -A[k]  *A[11],
							A[k]  *A[7] -A[k+4]*A[3]};
						Ainv[j] = A3inv[3*i] * cross[3] +
							A3inv[3*i+1] * cross[4] +
							A3inv[3*i+2] * cross[5];
						Ainv[k] =-A3inv[3*i] * cross[0] -
							A3inv[3*i+1] * cross[1] -
							A3inv[3*i+2] * cross[2];
						Ainv[3] = Ainv[j] + Ainv[k];
					} else {
					// starting point is on the segment
						Ainv[j] = A[k]*e[3*i] + A[k+4]*e[3*i+1] + A[k+8]*e[3*i+2];
						Ainv[k] =-A[j]*e[3*i] - A[j+4]*e[3*i+1] - A[j+8]*e[3*i+2];
						Ainv[3] = l[i];
					}
					if(coeff != NULL) {
						if(Ainv[3] >=-precision && Ainv[3] <= precision)
							Ainv[3] = precision;
						coeff[i] = 0;
						coeff[j] = Ainv[j] / Ainv[3];
						coeff[k] = Ainv[k] / Ainv[3];
//						coeff[3] = norm_ / Ainv[3];
					}
					return	Ainv[i] >=-precision && Ainv[i] <= precision &&
						Ainv[j] >=-precision &&
						Ainv[k] >=-precision &&
						(both_direction || Ainv[3] > precision);
				}
			} else {
			// direction parallel to triangle
				Ainv[0] = A3inv[0]*S[0]+A3inv[1]*S[1]+A3inv[2]*S[2];
				Ainv[1] = A3inv[3]*S[0]+A3inv[4]*S[1]+A3inv[5]*S[2];
				Ainv[2] = A3inv[6]*S[0]+A3inv[7]*S[1]+A3inv[8]*S[2];
				unsigned i = (Ainv[0] < Ainv[1] ? 0 : 1), j, k;
				i = (Ainv[i] < Ainv[2] ? i : 2);
				j = (i+1) % 3;
				k = (i+2) % 3;
				if(Ainv[k] < -precision) {
					k = j; j = i; i = 3 - j - k;
				}
				if(Ainv[j] < -precision) {
					scalar_t cross[] = {
						A[i+4]*A[11]-A[i+8]*A[7],
						A[i+8]*A[3] -A[i]  *A[11],
						A[i]  *A[7] -A[i+4]*A[3],
						A[j+4]*A[11]-A[j+8]*A[7],
						A[j+8]*A[3] -A[j]  *A[11],
						A[j]  *A[7] -A[j+4]*A[3],
						A[k+4]*A[11]-A[k+8]*A[7],
						A[k+8]*A[3] -A[k]  *A[11],
						A[k]  *A[7] -A[k+4]*A[3]};
					scalar_t dot[] = {
						A3inv[3*i]  * cross[6] +
						A3inv[3*i+1]* cross[7] +
						A3inv[3*i+2]* cross[8],
						-A3inv[3*i] * cross[3] -
						A3inv[3*i+1]* cross[4] -
						A3inv[3*i+2]* cross[5],

						A3inv[3*j]  * cross[0] +
						A3inv[3*j+1]* cross[1] +
						A3inv[3*j+2]* cross[2],
						-A3inv[3*j] * cross[6] -
						A3inv[3*j+1]* cross[7] -
						A3inv[3*j+2]* cross[8]};
					scalar_t sum[] = {dot[0]+dot[1], dot[2]+dot[3]};
					scalar_t norm[]= {
						A3inv[3*i]  * A3inv[3*i]  +
						A3inv[3*i+1]* A3inv[3*i+1]+
						A3inv[3*i+2]* A3inv[3*i+2],
						A3inv[3*j]  * A3inv[3*j]  +
						A3inv[3*j+1]* A3inv[3*j+1]+
						A3inv[3*j+2]* A3inv[3*j+2]};
					bool valid[] = {
						dot[0] >=-precision && dot[1] >=-precision &&
						(both_direction || norm[0] > precision),
						dot[2] >=-precision && dot[3] >=-precision &&
						(both_direction || norm[1] > precision)};
					if(coeff != NULL) {
						if(valid[0]) {
							coeff[i] = 0;
							coeff[j] = dot[0] / sum[0];
							coeff[k] = dot[1] / sum[0];
//							coeff[3] = norm[0] / sum[0];
						} else {
							coeff[i] = dot[3] / sum[1];
							coeff[j] = 0;
							coeff[k] = dot[2] / sum[1];
//							coeff[3] = norm[1] / sum[1];
						}
					}
					return (valid[0] || valid[1]) &&
						Ainv[3] >=-precision && Ainv[3] <= precision;
				} else if(Ainv[i] < -precision) {
					scalar_t cross[] = {
						A[j+4]*A[11]-A[j+8]*A[7],
						A[j+8]*A[3] -A[j]  *A[11],
						A[j]  *A[7] -A[j+4]*A[3],
						A[k+4]*A[11]-A[k+8]*A[7],
						A[k+8]*A[3] -A[k]  *A[11],
						A[k]  *A[7] -A[k+4]*A[3]};
					Ainv[j] = A3inv[3*i] * cross[3] +
						A3inv[3*i+1] * cross[4] +
						A3inv[3*i+2] * cross[5];
					Ainv[k] =-A3inv[3*i] * cross[0] -
						A3inv[3*i+1] * cross[1] -
						A3inv[3*i+2] * cross[2];
					Ainv[i] = Ainv[j] + Ainv[k];
					// scalar_t norm_ =
					// 	A3inv[3*i]  * A3inv[3*i]  +
					// 	A3inv[3*i+1]* A3inv[3*i+1]+
					// 	A3inv[3*i+2]* A3inv[3*i+2];
					if(coeff != NULL) {
						if(Ainv[i] >=-precision && Ainv[i] <= precision)
							Ainv[i] = precision;
						coeff[i] = 0;
						coeff[j] = Ainv[j] / Ainv[i];
						coeff[k] = Ainv[k] / Ainv[i];
//						coeff[3] = norm_ / Ainv[i];
					}
					return	Ainv[j] >=-precision &&
						Ainv[k] >=-precision &&
						Ainv[3] >=-precision && Ainv[3] <= precision &&
						(both_direction || Ainv[i] > precision);
				} else if(coeff != NULL) {
					coeff[0] = Ainv[0] / area;
					coeff[1] = Ainv[1] / area;
					coeff[2] = Ainv[2] / area;
//					coeff[3] = 0;
				}
				return	Ainv[i] >=-precision &&
					Ainv[3] >=-precision && Ainv[3] <= precision;
			}
		}
	}
}


template<typename scalar_t, typename index, unsigned char dim>
__global__ void search_ray_grid_kernel(
		const index *tri_num, const index *tri_idx,
		const index *size, const scalar_t *_min, scalar_t step,
		const scalar_t *points_base, const index *tri,
		const scalar_t *_origin, const scalar_t *_direction,
		bool *_valid, index points_num,
		scalar_t *coeff = NULL, index exclude_ind = 0,
		bool both_dir = false, scalar_t max_r2 = 0) {
	// const unsigned char dim = 3;
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	const scalar_t precision = 1e-9;
	if(points_base == NULL || tri == NULL || _origin == NULL || _direction == NULL
	|| tri_num == NULL || tri_idx == NULL || size == NULL || _min == NULL
	|| step <= 0 || id >= points_num || _valid == NULL)
		return;
	bool *valid = _valid + id;
	const scalar_t *origin = _origin + id * 3;
	const scalar_t *direction = _direction + id * 3;
	index	inter_ind = tri_num[size[dim]-1], x[dim*2+2];
	scalar_t	inter_point[dim], _coeff[dim+1],
		direction_[] = {-direction[0],-direction[1],-direction[2]},
		dist2 = 0, e = 0, dis2 = (max_r2 <= 0 ? -1 : max_r2);
	unsigned char out_dim[2] = {0, 2 * dim};
	for(unsigned char d = 0; d < dim; ++d) {
		dist2 += direction[d] * direction[d];
		scalar_t xf = (origin[d] - _min[d]) / step;
		if(xf < 0 || xf >= size[d]) {
			x[dim] = size[dim]; break;
		}
		x[dim+1+d] = x[d] = (index)xf;
		x[dim] = d > 0 ? x[dim] * size[d] + x[d] : x[d];
	}
	if(dist2 < precision) {
		valid[0] = (inter_ind != tri_num[size[dim]-1]);
		return;
	}
	if(x[dim] >= size[dim]) {
		out_dim[0] = ray_intersect_grid<scalar_t,index,dim>(origin, direction,
			step, _min, size, size[dim], true, inter_point);
		if(out_dim[0] >= 2 * dim && !both_dir){
			valid[0] = false;
			return;
		}
		for(unsigned char d = 0; d < dim; ++d) {
			scalar_t xf = (inter_point[d] - _min[d]) / step;
			xf = (xf < 0 ? 0 :(xf >= size[d] ? size[d]-1 : floor(xf)));
			x[d] = (index)xf;
			x[dim] = d > 0 ? x[dim] * size[d] + x[d] : x[d];
		}
		if(both_dir) {
			out_dim[1] = ray_intersect_grid<scalar_t,index,dim>(origin,direction_,
				step, _min, size, size[dim], true, inter_point);
			if(out_dim[1] < 2 * dim) {
				for(unsigned char d = 0; d < dim; ++d) {
					scalar_t xf = (inter_point[d] - _min[d]) / step;
					xf = (xf < 0 ? 0 :(xf >= size[d]?size[d]-1:floor(xf)));
					x[dim+1+d] = (index)xf;
					x[dim+1+dim] = d > 0 ?
						x[dim*2+1]*size[d] + x[dim+1+d] : x[dim+1+d];
				}
			} else if(out_dim[0] >= 2 * dim){
				valid[0] = (inter_ind != tri_num[size[dim]-1]);
				return;
			}
		}
	} else if(both_dir) {
		out_dim[1] = 0;
		x[dim*2+1] = x[dim];
	}
	while(out_dim[0] < 2 * dim || out_dim[1] < 2 * dim) {
		for(index j = (x[dim]==0?0:tri_num[x[dim]-1]); j < tri_num[x[dim]]; ++j) {
			if(exclude_ind > 0) {
				if(tri[dim*tri_idx[j]-3] == exclude_ind-1
				|| tri[dim*tri_idx[j]-2] == exclude_ind-1
				|| tri[dim*tri_idx[j]-1] == exclude_ind-1)
					continue;
			} else if(exclude_ind+1+tri_idx[j] == 0)
				continue;
			if(intersect_tri2<scalar_t>(origin, direction,
			points_base + dim * tri[dim * tri_idx[j] - 3],
			points_base + dim * tri[dim * tri_idx[j] - 2],
			points_base + dim * tri[dim * tri_idx[j] - 1],
			_coeff, false, precision)) {
				dist2 = 0;
				for(unsigned char d = 0; d < dim; ++d) {
					inter_point[d] =
						_coeff[0]*points_base[d+dim*tri[dim*tri_idx[j]-3]]+
						_coeff[1]*points_base[d+dim*tri[dim*tri_idx[j]-2]]+
						_coeff[2]*points_base[d+dim*tri[dim*tri_idx[j]-1]];
					e = inter_point[d] - origin[d];
					dist2 += e * e;
				}
				out_dim[0] = 2 * dim;
				if(dis2 < 0 || dist2 < dis2) {
					if(coeff != NULL)
						for(unsigned char d = 0; d < dim; ++d)
							coeff[d] = _coeff[d];
					inter_ind = tri_idx[j] - 1;
					dist2 = dis2;
				}
			}
		}
		if(out_dim[1] < 2 * dim) {
			for(index j = (x[dim*2+1]==0?0:tri_num[x[dim*2+1]-1]);
			j < tri_num[x[dim*2+1]]; ++j) {
				if(exclude_ind > 0) {
					if(tri[dim*tri_idx[j]-3] == exclude_ind-1
					|| tri[dim*tri_idx[j]-2] == exclude_ind-1
					|| tri[dim*tri_idx[j]-1] == exclude_ind-1)
						continue;
				}
				if(intersect_tri2<scalar_t>(origin, direction_,
				points_base + dim * tri[dim * tri_idx[j] - 3],
				points_base + dim * tri[dim * tri_idx[j] - 2],
				points_base + dim * tri[dim * tri_idx[j] - 1],
				_coeff, false, precision)) {
					dist2 = 0;
					for(unsigned char d = 0; d < dim; ++d) {
						inter_point[d] =
						_coeff[0]*points_base[d+dim*tri[dim*tri_idx[j]-3]]+
						_coeff[1]*points_base[d+dim*tri[dim*tri_idx[j]-2]]+
						_coeff[2]*points_base[d+dim*tri[dim*tri_idx[j]-1]];
						e = inter_point[d] - origin[d];
						dist2 += e * e;
					}
					out_dim[1] = 2 * dim;
					if(dis2 < 0 || dist2 < dis2) {
						if(coeff != NULL)
							for(unsigned char d = 0; d < dim; ++d)
								coeff[d] = _coeff[d];
						inter_ind = tri_idx[j] - 1;
						dist2 = dis2;
					}
				}
			}
			if(out_dim[1] < 2 * dim) {
				out_dim[1] = ray_intersect_grid<scalar_t,index,dim>(origin,direction_,
					step, _min, size, x[dim*2+1], false, inter_point);
				if(dis2 >= 0) {
					dist2 = 0;
					for(unsigned char d = 0; d < dim; ++d) {
						e = inter_point[d] - origin[d];
						dist2 += e * e;
					}
					if(dist2 > dis2)
						out_dim[1] = 2 * dim;
				}
				if(out_dim[1] < 2 * dim) {
					if(out_dim[1] % 2 == 1) {
						if(x[dim+1+out_dim[1]/2] == size[out_dim[1]/2] - 1)
							out_dim[1] = 2 * dim;
						else
							++x[dim+1+out_dim[1]/2];
					} else { if(x[dim+1+out_dim[1]/2] == 0)
							out_dim[1] = 2 * dim;
						else
							--x[dim+1+out_dim[1]/2];
					}
				}
			}
			if(out_dim[1] < 2 * dim) {
				for(unsigned char d = 0; d < dim; ++d)
					x[dim*2+1] = d > 0 ?
						x[dim*2+1]*size[d]+x[dim+1+d]:x[dim+1+d];
			} else if(out_dim[0] >= 2 * dim) {
				valid[0] = (inter_ind != tri_num[size[dim]-1]);
				return;
			}
		} else if(out_dim[0] >= 2 * dim){
			valid[0] = (inter_ind != tri_num[size[dim]-1]);
			return;
		}
		out_dim[0] = ray_intersect_grid<scalar_t,index,dim>(origin, direction,
			step, _min, size, x[dim], false, inter_point);
		if(dis2 >= 0) {
			dist2 = 0;
			for(unsigned char d = 0; d < dim; ++d) {
				e = inter_point[d] - origin[d];
				dist2 += e * e;
			}
			if(dist2 > dis2)
				out_dim[0] = 2 * dim;
		}
		if(out_dim[0] < 2 * dim) {
			if(out_dim[0] % 2 == 1) {
				if(x[out_dim[0]/2] == size[out_dim[0]/2] - 1)
					out_dim[0] = 2 * dim;
				else
					++x[out_dim[0]/2];
			} else { if(x[out_dim[0]/2] == 0)
					out_dim[0] = 2 * dim;
				else
					--x[out_dim[0]/2];
			}
			if(out_dim[0] < 2 * dim)
				for(unsigned char d = 0; d < dim; ++d)
					x[dim] = d > 0 ? x[dim]*size[d]+x[d] : x[d];
		}
	}
	valid[0] = (inter_ind != tri_num[size[dim]-1]);
	return;
}

void search_intersect_cuda (
	at::Tensor origins,
	at::Tensor directions,
	at::Tensor verts,
	at::Tensor faces,
	at::Tensor tri_num,
	at::Tensor tri_idx,
	at::Tensor num,
	at::Tensor minmax,
	float step,
	at::Tensor intersect
) {
	if(origins.sizes().size() != 2) origins = origins.reshape({-1,3});
	if(directions.sizes().size() != 2) directions = directions.reshape({-1,3});
	int32_t points_num = origins.size(0);

	const int threads = 512;
	const dim3 blocks (points_num / threads + 1, 1, 1);

	// make output
	// intersect.resize_({points_num});
	// intersect.zero_();

	AT_DISPATCH_FLOATING_TYPES(verts.type(), "search_intersect_cuda", ([&] {
		search_ray_grid_kernel<scalar_t, int32_t, 3><<<blocks, threads>>>(
			tri_num.data<int32_t>(),
			tri_idx.data<int32_t>(),
			num.data<int32_t>(),
			minmax.data<scalar_t>(),
			step,
			verts.data<scalar_t>(),
			faces.data<int32_t>(),
			origins.data<scalar_t>(),
			directions.data<scalar_t>(),
			intersect.data<bool>(),
			points_num
		);
	}));
	// __global__ void search_ray_grid_kernel(
	// 	const index *tri_num, const index *tri_idx,
	// 	const index *size, const scalar_t *_min, scalar_t step,
	// 	const scalar_t *points_base, const index *tri,
	// 	const scalar_t *_origin, const scalar_t *_direction,
	// 	bool *_valid, index points_num,
	// 	scalar_t *coeff = NULL, index exclude_ind = 0,
	// 	bool both_dir = false, scalar_t max_r2 = 0)

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
			printf("Error in search_intersect_cuda: %s\n", cudaGetErrorString(err));

}

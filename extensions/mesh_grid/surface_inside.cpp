#define USE_CUDA
#ifdef USE_CUDA
#include <stdint.h>
#include <string.h>
template<typename scalar,typename index>
extern scalar surface_inside_integral(unsigned char,index,
	const scalar*,const index*,const scalar*,scalar*,scalar=1e-6);
template<typename scalar, typename index>
extern bool surface_inside_gpu(index,index,index,char*,
	const scalar*,const scalar*,const index*,scalar=1e-6,
	const scalar* =NULL,const index* =NULL,const index* =NULL,const index* =NULL);
template<typename scalar, typename index>
extern scalar surface_inside_grid(unsigned char,index,const scalar*,
	const index*,const scalar*,scalar*,const scalar*,const index*,
	const index*,const index*,index = 256);
#else
#include "surface_inside.h"
#endif
#include "torch_util.h"
template<typename scalar, typename index>
index surface_inside_cpu(index n, index d, index m, char *inside,
			const scalar *points, const scalar *v,
			const index *tri, scalar eps = 1e-6,
			const scalar *_min_step = NULL, const index *size = NULL,
			const index *tri_num = NULL, const index *tri_idx = NULL) {
	bool has_grid =(_min_step != NULL && size != NULL &&
			tri_num != NULL && tri_idx != NULL);
	index num = 0;
	eps = (eps < 0 ? -eps : eps);
	scalar *patch = (scalar*)malloc(sizeof(scalar) * d * d);
	if(patch == NULL) return 0;
	if(has_grid) {
		for(index i = 0; i < n; ++i) {
			scalar	r = surface_inside_grid<scalar,index>(
					d, m, v, tri, points + d*i, patch,
					_min_step, size, tri_num, tri_idx);
			if(inside != NULL) {
				if((r - floor(r)) <= eps) {
					inside[i] = ((index)floor(r < 0 ? -r : r) % 2);
					num += inside[i];
				} else	inside[i] = -1; // on the boundary
			}
		}
	} else	for(index i = 0; i < n; ++i) {
			scalar	r = surface_inside_integral<scalar,index>(
					d, m, v, tri, points + d*i, patch, eps);
			if(inside != NULL) {
				if((r - floor(r)) <= eps) {
					inside[i] = ((index)floor(r < 0 ? -r : r) % 2);
					num += inside[i];
				} else	inside[i] = -1; // on the boundary
			}
		}
	free(patch);
	return	num;
}
using namespace std;
using namespace torch;
torch::Tensor surface_inside(torch::Tensor points,
		torch::Tensor vertices, torch::Tensor tri,
		torch::Tensor params,  torch::Tensor tri_num,
		torch::Tensor tri_idx, double eps = 1e-6) {
	int64_t	n = get_size(points, 0),
		d = get_size(points, 1),
		m = get_size(tri, 0);
	bool	isCuda = points.type().is_cuda(),
		has_grid = false;
	vector<int64_t> sz = {n, d};
	CHECK_SIZE(points, sz);
	sz[0] = get_size(vertices, 0);
	CHECK_SIZE(vertices, sz);
	CHECK_TYPE(points, vertices);
	sz[0] = m;
	CHECK_SIZE(tri, sz);
	CHECK_TYPE(tri, tri_num);
	sz = get_size(params);
	if(sz.size() == 1 && sz[0] == d + 1) {
		CHECK_TYPE(params, points);
		sz = get_size(tri_num);
		if(sz.size() == d) {
			vector<int64_t> s = get_size(tri_idx);
			if(s.size() == 1) {
				CHECK_TYPE(tri_num, tri_idx);
				has_grid = true;
				sz.push_back(1);
				for(unsigned char i = 0; i < d; ++i)
					sz[d] *= sz[i];
			}
		}
	}
	Tensor	inside = torch::zeros({n}, NEW_TYPE(kChar,isCuda));
	char  *inside_ = (char*)inside.data_ptr();
	switch(TYPE(points)) {
	case ScalarType::Float:
		if(isCuda) {
#ifdef USE_CUDA
			surface_inside_gpu<float,int64_t>(n, d, m,
				inside_, points.data<float>(),
				vertices.data<float>(), tri.data<int64_t>(),
				(float)eps,
				has_grid ? params.data<float>() : NULL,
				has_grid ? sz.data() : NULL,
				has_grid ? tri_num.data<int64_t>() : NULL,
				has_grid ? tri_idx.data<int64_t>() : NULL);
#endif
		} else {
			surface_inside_cpu<float,int64_t>(n, d, m,
				inside_, points.data<float>(),
				vertices.data<float>(), tri.data<int64_t>(),
				(float)eps,
				has_grid ? params.data<float>() : NULL,
				has_grid ? sz.data() : NULL,
				has_grid ? tri_num.data<int64_t>() : NULL,
				has_grid ? tri_idx.data<int64_t>() : NULL);
		} break;
	case ScalarType::Double:
		if(isCuda) {
#ifdef USE_CUDA
			surface_inside_gpu<double,int64_t>(n, d, m,
				inside_, points.data<double>(),
				vertices.data<double>(), tri.data<int64_t>(), eps,
				has_grid ? params.data<double>() : NULL,
				has_grid ? sz.data() : NULL,
				has_grid ? tri_num.data<int64_t>() : NULL,
				has_grid ? tri_idx.data<int64_t>() : NULL);
#endif
		} else {
			surface_inside_cpu<double,int64_t>(n, d, m,
				inside_, points.data<double>(),
				vertices.data<double>(), tri.data<int64_t>(), eps,
				has_grid ? params.data<double>() : NULL,
				has_grid ? sz.data() : NULL,
				has_grid ? tri_num.data<int64_t>() : NULL,
				has_grid ? tri_idx.data<int64_t>() : NULL);
		} break;
	default: CHECK_FLOAT(points);}
	return	inside;
}
PYBIND11_MODULE(surface_inside, m) {
	m.def("forward", &surface_inside, "Point Inside Surface");
}

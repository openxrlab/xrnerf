#include <iostream>
#include <vector>
#include <limits>
#include <stdint.h>
#include <ATen/ATen.h>
#ifdef USE_CUDA
template<typename scalar,typename index,class vector>
index zbuffer_forward(index,index,index,index,const scalar*,const index*,
	scalar*,vector*,index*,scalar*,bool*,bool,scalar);
template<typename scalar,typename index>
bool zbuffer_forward_gpu(index,index,index,index,const scalar*,const index*,
	index*,scalar*, bool*,bool,scalar);
#else
#include "render.h"
#endif
#include <torch/extension.h>
using namespace torch;
template<typename scalar,typename index>
index zbuffer_forward_cpu(index h, index w, index n, index f,
		const scalar *v, const index *tri,
		index *ind, scalar *coeff, bool *vis,
		bool persp, scalar eps) {
	scalar	*zbuf = (scalar*)malloc(sizeof(scalar)*h*w);
	std::vector<std::vector<index> > ibuf(h*w);
	for(index i = 0; i < h*w; ++i) {
		zbuf[i] = std::numeric_limits<scalar>::max();
		ibuf[i].clear();
	}
	index r = zbuffer_forward<scalar,index,std::vector<index> >
		(h, w, n, f, v, tri, zbuf, ibuf.data(), ind, coeff, vis, persp, eps);
	free(zbuf);
	return r;
}
std::vector<Tensor> render_forward(Tensor verts, Tensor tri,
			uint64_t h, uint64_t w,
			bool persp, double eps = 1e-6) {
	uint64_t n = verts.size(0),
		 f = tri.size(0);
	bool	cuda = verts.type().is_cuda();
	Tensor	index =-torch::ones({(int64_t)h,(int64_t)w}, cuda ? CUDA(kLong) : CPU(kLong)),
		visual= torch::ones({(int64_t)n}, cuda ? CUDA(kBool) : CPU(kBool)),
		coeff;
	switch(verts.type().scalarType()) {
	case torch::ScalarType::Float:
		coeff = torch::zeros({(int64_t)h,(int64_t)w,3}, cuda ? CUDA(kFloat) : CPU(kFloat));
		if(cuda) {
#ifdef USE_CUDA
			zbuffer_forward_gpu<float,int64_t>(
				(int64_t)h,(int64_t)w,(int64_t)n,(int64_t)f,
				verts.data<float>(), tri.data<int64_t>(),
				index.data<int64_t>(),coeff.data<float>(),visual.data<bool>(),
				persp, (float)eps);
#endif
		} else {
			zbuffer_forward_cpu<float,int64_t>(
				(int64_t)h,(int64_t)w,(int64_t)n,(int64_t)f,
				verts.data<float>(), tri.data<int64_t>(),
				index.data<int64_t>(),coeff.data<float>(),visual.data<bool>(),
				persp, (float)eps);
		} break;
	default:  break;}
	return {index, coeff, visual};
}
PYBIND11_MODULE(_render, m) {
	m.def("forward", &render_forward);
}

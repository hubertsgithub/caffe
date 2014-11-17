#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanMaskedLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  bool norm_with_count = this->layer_param_.euclidean_param().norm_with_count();
  float normwith = norm_with_count ? count : bottom[0]->num();

  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  // The 3rd bottom is the mask
  // If the mask is 0, 1, the 0 elements will be masked out
  caffe_gpu_mul(count,
  		  diff_.gpu_data(),
  		  bottom[2]->gpu_data(),
  		  diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / normwith / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanMaskedLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  bool norm_with_count = this->layer_param_.euclidean_param().norm_with_count();
  int count = bottom[0]->count();

  // Do not compute back propagation for the mask layer!
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
	  float normwith = norm_with_count ? bottom[i]->count() : bottom[i]->num();
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / normwith;
      caffe_gpu_axpby(
          count,              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
	  // The 3rd bottom is the mask
	  // If the mask is 0, 1, the 0 elements will be masked out
	  caffe_gpu_mul(count,
			  bottom[i]->gpu_diff(),
			  bottom[2]->gpu_data(),
			  bottom[i]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanMaskedLossLayer);

}  // namespace caffe

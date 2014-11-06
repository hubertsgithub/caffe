#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanMaskedLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  // 0: result from CNN, 1: label to compare with, 2: mask
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());

  CHECK_EQ(bottom[1]->height(), bottom[2]->height());
  CHECK_EQ(bottom[1]->width(), bottom[2]->width());
  CHECK_EQ(bottom[1]->channels(), bottom[2]->channels());

  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void EuclideanMaskedLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  // The 3rd bottom is the mask
  // If the mask is 0, 1, the 0 elements will be masked out
  caffe_mul(count,
  		  diff_.cpu_data(),
  		  bottom[2]->cpu_data(),
  		  diff_.mutable_cpu_data());

  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanMaskedLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  // Do not compute back propagation for the mask layer!
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          count,              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
	  // The 3rd bottom is the mask
	  // If the mask is 0, 1, the 0 elements will be masked out
	  caffe_mul(count,
			  bottom[i]->cpu_diff(),
			  bottom[2]->cpu_data(),
			  bottom[i]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanMaskedLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanMaskedLossLayer);
REGISTER_LAYER_CLASS(EUCLIDEAN_MASKED_LOSS, EuclideanMaskedLossLayer);
}  // namespace caffe

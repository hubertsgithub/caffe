#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void ReshapeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    ReshapeParameter reshape_param = this->layer_param_.reshape_param();
    int num = reshape_param.num();
    int channels = reshape_param.channels();
    int height = reshape_param.height();
    int width = reshape_param.width();

    if (num == 0)
        num = bottom[0]->num();
    if (channels == 0)
        channels = bottom[0]->channels();
    if (height == 0)
        height = bottom[0]->height();
    if (width == 0)
        width = bottom[0]->width();

    CHECK_EQ(bottom[0]->count(), num * channels * height * width);
    top[0]->Reshape(num, channels, height, width);
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  top[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  bottom[0]->ShareDiff(*top[0]);
}


#ifdef CPU_ONLY
STUB_GPU(ReshapeLayer);
#endif

INSTANTIATE_CLASS(ReshapeLayer);
REGISTER_LAYER_CLASS(RESHAPE, ReshapeLayer);
}  // namespace caffe

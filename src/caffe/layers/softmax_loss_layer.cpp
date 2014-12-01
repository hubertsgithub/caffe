#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const int nlabels = prob_.channels();
  int num = prob_.num();
  int dim = prob_.count() / num;
  int spatial_dim = prob_.height() * prob_.width();
  double loss = 0;
  if (bottom.size() == 3) {
      // spatially weighted version
      const Dtype* spatial_weight = bottom[2]->cpu_data();
      // Compute the sum of the weight
      const Dtype weight_sum = bottom[2]->asum_data();

      CHECK_EQ(bottom[2]->num(), bottom[0]->num());
      CHECK_EQ(bottom[2]->channels(), 1);
      CHECK_EQ(bottom[2]->height(), bottom[0]->height());
      CHECK_EQ(bottom[2]->width(), bottom[0]->width());
      CHECK_EQ(bottom[2]->count(), num * spatial_dim);

      // same loss function, scaled by a spatial weight
      for (int i = 0; i < num; ++i) {
        for (int j = 0; j < spatial_dim; j++) {
          const int label_value = static_cast<int>(label[i * spatial_dim + j] + 0.5);
          CHECK_GE(label_value, 0);
          if (label_value < nlabels) {
              CHECK_GT(dim, label_value * spatial_dim);
              const double prob_ij = prob_data[i * dim + label_value * spatial_dim + j];
              const double loss_ij = log(std::max<double>(prob_ij, 1e-20));
              const double weight_ij = spatial_weight[i * spatial_dim + j];
              loss -= weight_ij * loss_ij;
          }
        }
      }
      CHECK(!std::isnan(loss));
      top[0]->mutable_cpu_data()[0] = loss / weight_sum / num;
  } else {
      // no spatial weights
      for (int i = 0; i < num; ++i) {
        for (int j = 0; j < spatial_dim; j++) {
          const int label_value = static_cast<int>(label[i * spatial_dim + j] + 0.5);
          CHECK_GE(label_value, 0);
          if (label_value < nlabels) {
              CHECK_GT(dim, label_value * spatial_dim);
              const double prob_ij = prob_data[i * dim + label_value * spatial_dim + j];
              const double loss_ij = log(std::max<double>(prob_ij, 1e-20));
              loss -= loss_ij;
          }
        }
      }
      CHECK(!std::isnan(loss));
      top[0]->mutable_cpu_data()[0] = loss / spatial_dim / num;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down.size() >= 3 && propagate_down[2]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to spatial weights.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    int spatial_dim = prob_.height() * prob_.width();
    int nlabels = prob_.channels();

    caffe_copy(prob_.count(), prob_data, bottom_diff);
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        const int label_value = static_cast<int>(label[i * spatial_dim + j] + 0.5);
        CHECK_GE(label_value, 0);
        CHECK_LE(label_value, nlabels);
        if (label_value < nlabels) {
            bottom_diff[i * dim + label_value * spatial_dim + j] -= 1;
        }
      }
    }

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (bottom.size() == 3) {
        // scale by spatial weight
        const Dtype* spatial_weight = bottom[2]->cpu_data();
		// Compute the sum of the weight
		const Dtype weight_sum = bottom[2]->asum_data();

        int idx = 0;
        for (int i = 0; i < num; ++i) {
            for (int c = 0; c < nlabels; ++c) {
                for (int j = 0; j < spatial_dim; ++j) {
                    bottom_diff[idx++] *= spatial_weight[i * spatial_dim + j];
                }
            }
        }
        // Scale gradient
        caffe_scal(prob_.count(), loss_weight / weight_sum / num, bottom_diff);
    } else {
        // Scale gradient
        caffe_scal(prob_.count(), loss_weight / spatial_dim / num, bottom_diff);
    }

  }
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SOFTMAX_LOSS, SoftmaxWithLossLayer);
}  // namespace caffe

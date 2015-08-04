#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithTagLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  normalize_ = this->layer_param_.loss_param().normalize();
}

template <typename Dtype>
void SoftmaxWithTagLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Number of input tags must match number of predictions; "
      << "e.g., if prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*C*H*W, "
      << "with integer values in {0, 1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithTagLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int class_num = prob_.shape(softmax_axis_);
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
	for (int j = 0; j < inner_num_; j++) {
		Dtype closs = 0;
		int ccount = 0;
		for (int label_value = 0; label_value < class_num; ++label_value) {
			const int ind = i * dim + label_value * inner_num_ + j;
			const int tag = static_cast<int>(label[ind]);
			DCHECK_GE(tag, 0);
			DCHECK_LT(tag, 1);
			if (tag == 1) {
				closs -= log(std::max(prob_data[ind], Dtype(FLT_MIN)));
				++ccount;
			}
		}
		if (ccount == 0) continue;
		loss += closs / ccount;
	}
  }
  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = loss / (outer_num_ * inner_num_);
  } else {
    top[0]->mutable_cpu_data()[0] = loss / outer_num_;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithTagLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int class_num = prob_.shape(softmax_axis_);
    int dim = prob_.count() / outer_num_;
	for (int i = 0; i < outer_num_; ++i) {
		for (int j = 0; j < inner_num_; j++) {
			int ccount = 0;
			// First count how many attributes we have for this item
			for (int label_value = 0; label_value < class_num; ++label_value) {
				const int ind = i * dim + label_value * inner_num_ + j;
				const int tag = static_cast<int>(label[ind]);
				DCHECK_GE(tag, 0);
				DCHECK_LT(tag, 1);
				if (tag == 1) {
					++ccount;
				}
			}
			for (int label_value = 0; label_value < class_num; ++label_value) {
				const int ind = i * dim + label_value * inner_num_ + j;
				const int tag = static_cast<int>(label[ind]);
				DCHECK_GE(tag, 0);
				DCHECK_LT(tag, 1);
				if (tag == 1) {
					bottom_diff[ind] -= 1.0 / ccount;
				} else if (ccount == 0) {
					bottom_diff[ind] = 0;
				}
			}
        }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      caffe_scal(prob_.count(), loss_weight / (outer_num_ * inner_num_), bottom_diff);
    } else {
      caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithTagLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithTagLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithTagLoss);

}  // namespace caffe

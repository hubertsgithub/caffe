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
  if (bottom.size() > 2) {
    CHECK_EQ(bottom[0]->count() / (outer_num_ * inner_num_), bottom[2]->count())
        << "Number of tag frequencies must match length of softmax axis (the number of tags).";
  }
  // Reshape weights to the size equal to the number of possible different tags
  vector<int> weights_shape(4, 1);
  weights_shape[softmax_axis_] = prob_.shape(softmax_axis_);
  weights_.Reshape(weights_shape);
    
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithTagLossLayer<Dtype>::SetupWeights_cpu(const vector<Blob<Dtype>*>& bottom) {
  Dtype* weights_ptr = weights_.mutable_cpu_data();
  int class_num = prob_.shape(softmax_axis_);
  if (bottom.size() > 2) {
    // Compute reciprocal frequencies, use the diff array of weights_ as a temporary store, since it's not used anyway...
    caffe_set(class_num, (Dtype)1, weights_.mutable_cpu_diff());
    caffe_div(
      class_num, 
      weights_.cpu_diff(), 
      bottom[2]->cpu_data(),
      weights_ptr
    );
    // We want the weights to sum up to class_num, because if all weights are 1, we get back the non-weighted version
    Dtype scale = class_num / caffe_cpu_asum(class_num, weights_ptr);
    caffe_scal(class_num, scale, weights_ptr);
  } else {
    caffe_set(class_num, (Dtype)1, weights_ptr);
  }
}

template <typename Dtype>
void SoftmaxWithTagLossLayer<Dtype>::SetupWeights_gpu(const vector<Blob<Dtype>*>& bottom) {
  Dtype* weights_ptr = weights_.mutable_gpu_data();
  int class_num = prob_.shape(softmax_axis_);
  if (bottom.size() > 2) {
    // Compute reciprocal frequencies, use the diff array of weights_ as a temporary store, since it's not used anyway...
    caffe_gpu_set(class_num, (Dtype)1, weights_.mutable_gpu_diff());
    caffe_gpu_div(
      class_num, 
      weights_.gpu_diff(), 
      bottom[2]->gpu_data(),
      weights_ptr
    );
    // We want the weights to sum up to class_num, because if all weights are 1, we get back the non-weighted version
    Dtype scale;
    caffe_gpu_asum(class_num, weights_ptr, &scale);
    scale = class_num / scale;
    caffe_gpu_scal(class_num, scale, weights_ptr);
  } else {
    caffe_gpu_set(class_num, (Dtype)1, weights_ptr);
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
  SetupWeights_cpu(bottom);
  const Dtype* weights_ptr = weights_.cpu_data();

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
				closs -= log(std::max(prob_data[ind], Dtype(FLT_MIN))) * weights_ptr[label_value];
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
  if (propagate_down.size() > 2 && propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label frequency inputs.";
  }

  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int class_num = prob_.shape(softmax_axis_);
    int dim = prob_.count() / outer_num_;
    SetupWeights_cpu(bottom);
    const Dtype* weights_ptr = weights_.cpu_data();
	for (int i = 0; i < outer_num_; ++i) {
		for (int j = 0; j < inner_num_; j++) {
			int ccount = 0;
			Dtype sum_weights = 0;
			// First count how many attributes we have for this item
			for (int label_value = 0; label_value < class_num; ++label_value) {
				const int ind = i * dim + label_value * inner_num_ + j;
				const int tag = static_cast<int>(label[ind]);
				DCHECK_GE(tag, 0);
				DCHECK_LT(tag, 1);
				if (tag == 1) {
					++ccount;
					sum_weights += weights_ptr[label_value];
				}
			}
			for (int label_value = 0; label_value < class_num; ++label_value) {
				const int ind = i * dim + label_value * inner_num_ + j;
				const int tag = static_cast<int>(label[ind]);
				DCHECK_GE(tag, 0);
				DCHECK_LT(tag, 1);
				bottom_diff[ind] *= sum_weights / ccount;
				if (tag == 1) {
					bottom_diff[ind] -= 1.0 / ccount * weights_ptr[label_value];
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

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxTagLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int dim, const int inner_num, const int outer_num,
          const int class_num) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / inner_num;
    const int s = index % inner_num;
    int ccount = 0;
	// First count how many attributes we have for this item
	for (int label_value = 0; label_value < class_num; ++label_value) {
		const int ind = n * dim + label_value * inner_num + s;
		const int tag = static_cast<int>(label[ind]);

		if (tag == 1) {
			loss[ind] = -log(max(prob_data[ind], Dtype(FLT_MIN)));
			++ccount;
		}
	}
	if (ccount == 0) return;
	for (int label_value = 0; label_value < class_num; ++label_value) {
		const int ind = n * dim + label_value * inner_num + s;
		const int tag = static_cast<int>(label[ind]);

		if (tag == 1) {
			loss[ind] /= ccount;
		}
	}
  }
}

template <typename Dtype>
void SoftmaxWithTagLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  const int class_num = prob_.shape(softmax_axis_);
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxTagLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
      dim, inner_num_, outer_num_, class_num);
  Dtype loss;
  caffe_gpu_asum(prob_.count(), loss_data, &loss);
  if (normalize_) {
    loss /= outer_num_ * inner_num_;
  } else {
    loss /= outer_num_;
  }
  top[0]->mutable_cpu_data()[0] = loss;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void SoftmaxTagLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff,
          const int dim, const int inner_num, const int outer_num,
          const int class_num) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / inner_num;
    const int s = index % inner_num;
    int ccount = 0;
	// First count how many attributes we have for this item
	for (int label_value = 0; label_value < class_num; ++label_value) {
		const int ind = n * dim + label_value * inner_num + s;
		const int tag = static_cast<int>(label[ind]);

		if (tag == 1) {
			++ccount;
		}
	}
	for (int label_value = 0; label_value < class_num; ++label_value) {
		const int ind = n * dim + label_value * inner_num + s;
		const int tag = static_cast<int>(label[ind]);

		if (tag == 1) {
			bottom_diff[ind] -= 1.0 / ccount;
		} else if (ccount == 0) {
			bottom_diff[ind] = 0;
		}
	}
  }
}

template <typename Dtype>
void SoftmaxWithTagLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
	const int nthreads = outer_num_ * inner_num_;
	const int class_num = prob_.shape(softmax_axis_);
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxTagLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff,
        dim, inner_num_, outer_num_, class_num);
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      caffe_gpu_scal(
      		  prob_.count(), 
      		  loss_weight / (outer_num_ * inner_num_), 
      		  bottom_diff
	  );
    } else {
      caffe_gpu_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithTagLossLayer);

}  // namespace caffe

#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        top[1]->mutable_gpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}
INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

template <typename Dtype>
void BasePrefetchingMultiDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";

  // Copy the data
  CHECK_EQ(top.size(), this->prefetch_data_.size()) << "Mismatch in number of top/prefetch_label layers";
  for (int i = 0; i < this->prefetch_data_.size(); ++i) {
  	DLOG(INFO) << "Copying prefetched data #" << i;
  	caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
  		top[i]->mutable_gpu_data());
  }
  DLOG(INFO) << "Prefetch copied";

  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  CreatePrefetchThread();
}
INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingMultiDataLayer);

}  // namespace caffe

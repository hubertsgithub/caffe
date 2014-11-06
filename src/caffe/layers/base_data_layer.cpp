#include <string>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

////////////////////////////////////////
// BaseDataLayer
////////////////////////////////////////

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()),
      data_transformer_(transform_param_) {
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
  data_transformer_.InitRand();
}

////////////////////////////////////////
// BasePrefetchingDataLayer
////////////////////////////////////////

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  this->prefetch_data_.mutable_cpu_data();
  if (this->output_labels_) {
    this->prefetch_label_.mutable_cpu_data();
  }
  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread() {
  this->phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
               top[1]->mutable_cpu_data());
  }
  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  CreatePrefetchThread();
}

////////////////////////////////////////
// BasePrefetchingMultiDataLayer
////////////////////////////////////////

template <typename Dtype>
BasePrefetchingMultiDataLayer<Dtype>::BasePrefetchingMultiDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {

	MultiPrefetchDataParameter mpdp = param.multi_prefetch_data_param();
    int size = mpdp.data_transformations_size();
    for (int i = 0; i < size; ++i) {
    	TransformationParameter tp = mpdp.data_transformations(i);
    	this->transform_params_.push_back(tp);
    	this->transformers_.push_back(shared_ptr<DataTransformer<Dtype> >(new DataTransformer<Dtype>(tp)));
	}
}

template <typename Dtype>
void BasePrefetchingMultiDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top.size(), this->transformers_.size()) << "You have to specify a data transformer for each data in top";

  for (int i = 0; i < this->transformers_.size(); ++i) {
	  this->transformers_[i]->InitRand();
  }

  DLOG(INFO) << "Initializing prefetch and transformed data...";
  // Create the desired number of blobs for prefetch and transformed data
  this->prefetch_data_.resize(top.size());
  this->transformed_data_.resize(top.size());
  for (int i = 0; i < this->prefetch_data_.size(); ++i) {
	this->prefetch_data_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
	this->transformed_data_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  }

  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);

  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.

  for (int i = 0; i < this->prefetch_data_.size(); ++i) {
  	this->prefetch_data_[i]->mutable_cpu_data();
  }
  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingMultiDataLayer<Dtype>::CreatePrefetchThread() {
  this->phase_ = Caffe::phase();
  for (int i = 0; i < this->transformers_.size(); ++i) {
	  this->transformers_[i]->InitRand();
  }
  CHECK(StartInternalThread()) << "Thread execution failed";
}

template <typename Dtype>
void BasePrefetchingMultiDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

template <typename Dtype>
void BasePrefetchingMultiDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";

  // Copy the data
  CHECK_EQ(top.size(), this->prefetch_data_.size()) << "Mismatch in number of top/prefetch_label layers";
  for (int i = 0; i < this->prefetch_data_.size(); ++i) {
  	DLOG(INFO) << "Copying prefetched data #" << i;
  	caffe_copy(prefetch_data_[i]->count(), prefetch_data_[i]->cpu_data(),
  		top[i]->mutable_cpu_data());
  }
  DLOG(INFO) << "Prefetch copied";

  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  CreatePrefetchThread();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
STUB_GPU_FORWARD(BasePrefetchingMultiDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);
INSTANTIATE_CLASS(BasePrefetchingMultiDataLayer);

}  // namespace caffe

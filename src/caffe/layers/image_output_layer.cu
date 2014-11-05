#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ImageOutputLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
	  const int data_dim = bottom[i]->count() / bottom[i]->num();
  	  for (int n = 0; n < bottom[i].num(); ++n) {
		  cv::Mat cv_img = this->ConvertBlobToCVImg(*bottom[i], n, false);
		  stringstream ss;
		  ss << "imgsave-bottom" << i << "-batchid" << n << ".jpg";
		  WriteImageFromCVMat(ss.str());
		  LOG(INFO) << "Successfully saved " << data_blob_.num() << " rows";
	  }
  }
}

template <typename Dtype>
void ImageOutputLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

INSTANTIATE_LAYER_GPU_FUNCS(ImageOutputLayer);

}  // namespace caffe

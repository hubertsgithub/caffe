#ifndef CAFFE_MULTI_IMAGE_DATA_LAYER_HPP_
#define CAFFE_MULTI_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net from multiple image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MultiImageDataLayer : public BasePrefetchingDataLayerOLD<Dtype> {
 public:
  explicit MultiImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayerOLD<Dtype>(param) {}
  virtual ~MultiImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Skip(int n);

  //virtual inline LayerParameter_LayerType type() const {
    //return LayerParameter_LayerType_MULTI_IMAGE_DATA;
  //}
  virtual inline const char* type() const { return "MultiImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();
  virtual cv::Mat ToGrayscale(const cv::Mat& cv_img);

  struct patch_info {
    std::string path;
    float ax,ay,bx,by;
    int c;
  };
  struct line {
    std::vector<struct patch_info> image;
    float s,r,j;
    int label;
  };
  vector<struct line> lines_;
  vector<vector<int> > label_index_set_;
  vector<int> label_index_set_tail_;
  vector<int> unbalanced_index_set_;
  int unbalanced_index_set_tail_;
};

}  // namespace caffe

#endif  // CAFFE_MULTI_IMAGE_DATA_LAYER_HPP_

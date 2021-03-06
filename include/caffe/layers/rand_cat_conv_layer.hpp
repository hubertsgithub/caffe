#ifndef CAFFE_RAND_CAT_CONV_LAYER_HPP_
#define CAFFE_RAND_CAT_CONV_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Takes a Blob and crop it, to the shape specified by the second input
 *  Blob, across all dimensions after the specified axis.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
// added by ab --
// concat the points for different conv-layers
// randomly sample point from the images
// constraint -- all bottom blobs containing hypercol data
// should be consequential.
template <typename Dtype>
class RandCatConvLayer : public Layer<Dtype> {
 public:
  explicit RandCatConvLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RandCatConv"; }
  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  //shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 // no. of data points per image
 int N_;
 // if we need to run the random no. generator
 bool if_rand_;
 // if we need to do class balancing when random sampling
 bool if_balanced_;
 // Number of channels for the labels. E.g. it's 3 for normals, 1 for
 // antishadow labels.
 int label_channels_;
 // This determines the ratio when balancing between classes
 std::vector<float> class_balance_;
 // This should be equal to sum(class_balance_)
 int full_class_weight_;
 // This should be equal to class_balance_.size()
 int class_count_;
 // number of bottom blobs containing hypercol data
 int n_hblobs_;
 // bottom-blobs start and end ids
 int start_id_;
 int end_id_;
 // number of channels in the hypercol data --
 int n_channels_;
 // points which are randomly selected --
 std::vector<int> rand_points_;
 // pooling factor
 std::vector<int> poolf_;
 // padding factor
 std::vector<Dtype> padf_;
 // make a vector of height, width, num_pixels for diff conv.layers --
 std::vector<int> height_, width_, pixels_;

};


}  // namespace caffe

#endif  // CAFFE_RAND_CAT_CONV_LAYER_HPP_

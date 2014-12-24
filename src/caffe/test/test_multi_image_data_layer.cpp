#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class MultiImageDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MultiImageDataLayerTest()
      : seed_(1701),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()),
        blob_top_label2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    MakeTempFilename(&filename_);
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    blob_top_vec_.push_back(blob_top_label2_);
    Caffe::set_random_seed(seed_);
    // Create a Vector of files with label images
    std::ofstream outfile(filename_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_;
    for (int i = 0; i < 5; ++i) {
      outfile << EXAMPLES_SOURCE_DIR "images/cat.jpg " << EXAMPLES_SOURCE_DIR "images/fish-bike.jpg " << EXAMPLES_SOURCE_DIR "images/fish.jpg" << endl;
    }
    outfile.close();
  }

  virtual ~MultiImageDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
    delete blob_top_label2_;
  }

  int seed_;
  string filename_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  Blob<Dtype>* const blob_top_label2_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MultiImageDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(MultiImageDataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_shuffle(false);

  MultiPrefetchDataParameter* multi_prefetch_data_param = param.mutable_multi_prefetch_data_param();
  TransformationParameter* tp;
  tp = multi_prefetch_data_param->add_data_transformations();
  tp = multi_prefetch_data_param->add_data_transformations();
  tp = multi_prefetch_data_param->add_data_transformations();
  multi_prefetch_data_param->set_input_data_count(3);

  MultiImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 3);
  EXPECT_EQ(this->blob_top_label_->height(), 323);
  EXPECT_EQ(this->blob_top_label_->width(), 481);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label2_->channels(), 3);
  EXPECT_EQ(this->blob_top_label2_->height(), 279);
  EXPECT_EQ(this->blob_top_label2_->width(), 500);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Go through the data twice
  /*for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }*/
}

TYPED_TEST(MultiImageDataLayerTest, TestResize) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_shuffle(false);

  MultiPrefetchDataParameter* multi_prefetch_data_param = param.mutable_multi_prefetch_data_param();
  TransformationParameter* tp;
  tp = multi_prefetch_data_param->add_data_transformations();
  tp->set_new_height(256);
  tp->set_new_width(256);
  tp = multi_prefetch_data_param->add_data_transformations();
  tp->set_new_height(128);
  tp->set_new_width(100);
  tp = multi_prefetch_data_param->add_data_transformations();
  tp->set_new_height(128);
  tp->set_new_width(100);
  multi_prefetch_data_param->set_input_data_count(3);

  MultiImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 256);
  EXPECT_EQ(this->blob_top_data_->width(), 256);
  EXPECT_EQ(this->blob_top_label_->channels(), 3);
  EXPECT_EQ(this->blob_top_label_->height(), 128);
  EXPECT_EQ(this->blob_top_label_->width(), 100);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label2_->channels(), 3);
  EXPECT_EQ(this->blob_top_label2_->height(), 128);
  EXPECT_EQ(this->blob_top_label2_->width(), 100);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Go through the data twice
  /*for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }*/
}

TYPED_TEST(MultiImageDataLayerTest, TestCropFirst) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_shuffle(false);

  MultiPrefetchDataParameter* multi_prefetch_data_param = param.mutable_multi_prefetch_data_param();
  TransformationParameter* tp;
  tp = multi_prefetch_data_param->add_data_transformations();
  tp->set_new_height(256);
  tp->set_new_width(256);
  tp->set_crop_first(true);
  // 0 crop_size
  tp = multi_prefetch_data_param->add_data_transformations();
  tp->set_new_height(30);
  tp->set_new_width(60);
  tp->set_crop_first(true);
  tp->set_crop_size(50);
  tp = multi_prefetch_data_param->add_data_transformations();
  tp->set_new_height(128);
  tp->set_new_width(100);
  tp->set_crop_first(false);
  tp->set_crop_size(50);
  multi_prefetch_data_param->set_input_data_count(3);

  MultiImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 256);
  EXPECT_EQ(this->blob_top_data_->width(), 256);

  EXPECT_EQ(this->blob_top_label_->channels(), 3);
  EXPECT_EQ(this->blob_top_label_->height(), 30);
  EXPECT_EQ(this->blob_top_label_->width(), 60);
  EXPECT_EQ(this->blob_top_label_->num(), 5);

  EXPECT_EQ(this->blob_top_label2_->channels(), 3);
  EXPECT_EQ(this->blob_top_label2_->height(), 50);
  EXPECT_EQ(this->blob_top_label2_->width(), 50);

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Go through the data twice
  /*for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }*/
}

TYPED_TEST(MultiImageDataLayerTest, TestShuffle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_shuffle(true);

  MultiPrefetchDataParameter* multi_prefetch_data_param = param.mutable_multi_prefetch_data_param();
  TransformationParameter* tp;
  tp = multi_prefetch_data_param->add_data_transformations();
  tp = multi_prefetch_data_param->add_data_transformations();
  tp = multi_prefetch_data_param->add_data_transformations();
  multi_prefetch_data_param->set_input_data_count(3);

  MultiImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 3);
  EXPECT_EQ(this->blob_top_label_->height(), 323);
  EXPECT_EQ(this->blob_top_label_->width(), 481);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label2_->channels(), 3);
  EXPECT_EQ(this->blob_top_label2_->height(), 279);
  EXPECT_EQ(this->blob_top_label2_->width(), 500);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Go through the data twice
  /*for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    map<Dtype, int> values_to_indices;
    int num_in_order = 0;
    for (int i = 0; i < 5; ++i) {
      Dtype value = this->blob_top_label_->cpu_data()[i];
      // Check that the value has not been seen already (no duplicates).
      EXPECT_EQ(values_to_indices.find(value), values_to_indices.end());
      values_to_indices[value] = i;
      num_in_order += (value == Dtype(i));
    }
    EXPECT_EQ(5, values_to_indices.size());
    EXPECT_GT(5, num_in_order);
  }*/
}

}  // namespace caffe

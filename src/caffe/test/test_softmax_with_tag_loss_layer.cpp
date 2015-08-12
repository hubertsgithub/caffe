#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class SoftmaxWithTagLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SoftmaxWithTagLossLayerTest()
	  : blob_bottom_data_(new Blob<Dtype>(10, 5, 2, 3)),
		blob_bottom_label_(new Blob<Dtype>(10, 5, 2, 3)),
		blob_bottom_freqs_(new Blob<Dtype>(1, 5, 1, 1)),
        blob_bottom_label_softmax_(new Blob<Dtype>(10, 1, 2, 3)),
        blob_bottom_label_softmax_tag_(new Blob<Dtype>(10, 5, 2, 3)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_softmax_.push_back(blob_bottom_data_);
    blob_bottom_vec_softmax_tag_.push_back(blob_bottom_data_);

    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 2;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);

    for (int i = 0; i < blob_bottom_freqs_->count(); ++i) {
      // Make sure it's positive
      blob_bottom_freqs_->mutable_cpu_data()[i] = caffe_rng_rand() % 100 + 1;
    }
    blob_bottom_vec_.push_back(blob_bottom_freqs_);

    caffe_set(
    		blob_bottom_label_softmax_tag_->count(),
    		static_cast<Dtype>(0),
    		blob_bottom_label_softmax_tag_->mutable_cpu_data()
    );
	// Fill in the two label blobs with the "same" data
    for (int i = 0; i < blob_bottom_label_softmax_->shape(0); ++i) {
      for (int j = 0; j < blob_bottom_label_softmax_->shape(2); ++j) {
        for (int k = 0; k < blob_bottom_label_softmax_->shape(3); ++k) {
          Dtype r = caffe_rng_rand() % 5;
          int offset = blob_bottom_label_softmax_->offset(i, 0, j, k);
          blob_bottom_label_softmax_->mutable_cpu_data()[offset] = r;
          for (int l = 0; l < blob_bottom_label_softmax_tag_->shape(1); ++l) {
			int offset = blob_bottom_label_softmax_tag_->offset(i, l, j, k);
			if (l == r) {
			  blob_bottom_label_softmax_tag_->mutable_cpu_data()[offset] = 1;
			}
          }
        }
      }
    }
    blob_bottom_vec_softmax_.push_back(blob_bottom_label_softmax_);
    blob_bottom_vec_softmax_tag_.push_back(blob_bottom_label_softmax_tag_);

    blob_top_vec_.push_back(blob_top_loss_);
  }

  void testForwardCompare(bool normalize) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_loss_param()->set_normalize(normalize);
    // First, compute the loss with the original SoftmaxWithLossLayer
    scoped_ptr<SoftmaxWithLossLayer<Dtype> > layer(
        new SoftmaxWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_softmax_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_softmax_, this->blob_top_vec_);
    Dtype full_loss_softmax = this->blob_top_loss_->cpu_data()[0];

    // Then compute the loss with the new SoftmaxWithTagLossLayer
    scoped_ptr<SoftmaxWithTagLossLayer<Dtype> > layerTag(
        new SoftmaxWithTagLossLayer<Dtype>(layer_param));
    layerTag->SetUp(this->blob_bottom_vec_softmax_tag_, this->blob_top_vec_);
    layerTag->Forward(this->blob_bottom_vec_softmax_tag_, this->blob_top_vec_);
    Dtype full_loss_softmax_tag = this->blob_top_loss_->cpu_data()[0];
    // Check that the two losses are the same.
    EXPECT_NEAR(full_loss_softmax, full_loss_softmax_tag, 1e-4);
  }

  void testBackwardCompare(bool normalize) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_loss_param()->set_normalize(normalize);

	vector<bool> propagate_down(this->blob_bottom_vec_softmax_.size(), false);
	propagate_down[0] = true;

    // First, compute the loss with the original SoftmaxWithLossLayer
    scoped_ptr<SoftmaxWithLossLayer<Dtype> > layer(
        new SoftmaxWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_softmax_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_softmax_, this->blob_top_vec_);
    layer->Backward(
    		this->blob_top_vec_,
    		propagate_down,
    		this->blob_bottom_vec_softmax_
    );
    // Save gradient, because we will overwrite it later
	scoped_ptr<Blob<Dtype> > softmax_grad(new Blob<Dtype>());
	softmax_grad->CopyFrom(*this->blob_bottom_data_, true, true);

    // Then compute the loss with the new SoftmaxWithTagLossLayer
    scoped_ptr<SoftmaxWithTagLossLayer<Dtype> > layerTag(
        new SoftmaxWithTagLossLayer<Dtype>(layer_param));
    layerTag->SetUp(this->blob_bottom_vec_softmax_tag_, this->blob_top_vec_);
    layerTag->Forward(this->blob_bottom_vec_softmax_tag_, this->blob_top_vec_);
    layerTag->Backward(
    		this->blob_top_vec_,
    		propagate_down,
    		this->blob_bottom_vec_softmax_tag_
    );
    Dtype full_loss_softmax_tag = this->blob_top_loss_->cpu_data()[0];
    // Check that the two gradients are the same.
    for (int i = 0; i < this->blob_bottom_data_->count(); ++i) {
		EXPECT_NEAR(
				this->blob_bottom_data_->cpu_diff()[i],
				softmax_grad->cpu_diff()[i],
				1e-4
		);
	}
  }

  virtual ~SoftmaxWithTagLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_bottom_freqs_;
    delete blob_bottom_label_softmax_;
    delete blob_bottom_label_softmax_tag_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_freqs_;
  Blob<Dtype>* const blob_bottom_label_softmax_;
  Blob<Dtype>* const blob_bottom_label_softmax_tag_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_softmax_;
  vector<Blob<Dtype>*> blob_bottom_vec_softmax_tag_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxWithTagLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxWithTagLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  SoftmaxWithTagLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxWithTagLossLayerTest, TestForwardCompareNormalize) {
	this->testForwardCompare(true);
}

TYPED_TEST(SoftmaxWithTagLossLayerTest, TestForwardCompare) {
	this->testForwardCompare(false);
}

TYPED_TEST(SoftmaxWithTagLossLayerTest, TestBackwardCompareNormalize) {
	this->testBackwardCompare(true);
}

TYPED_TEST(SoftmaxWithTagLossLayerTest, TestBackwardCompare) {
	this->testBackwardCompare(false);
}

TYPED_TEST(SoftmaxWithTagLossLayerTest, TestGradientUnnormalized) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  SoftmaxWithTagLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void AccuracyMapLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();
}

template <typename Dtype>
void AccuracyMapLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[0]->num(), bottom[2]->num())
      << "The data and weights should have the same number.";
  CHECK_LE(top_k_, bottom[0]->channels())
      << "top_k must be less than or equal to the number of classes.";
  CHECK_GT(bottom[0]->channels(), 1)
      << "Must have at least 2 classes";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), bottom[2]->height());
  CHECK_EQ(bottom[0]->width(), bottom[2]->width());
  CHECK_EQ(bottom[1]->height(), bottom[2]->height());
  CHECK_EQ(bottom[1]->width(), bottom[2]->width());
  top[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void AccuracyMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* bottom_weight = bottom[2]->cpu_data();
  const int ni = bottom[0]->num();
  const int nc = bottom[0]->channels();
  const int nyx = bottom[0]->height() * bottom[0]->width();
  const int nyxc = nc * nyx;

  Dtype accuracy_sum = 0;

  for (int i = 0; i < ni; ++i) {
    for (int yx = 0; yx < nyx; ++yx) {
      // Top-k accuracy with spatial weighting
      const int true_label = static_cast<int>(bottom_label[i * nyx + yx]);
      const Dtype weight = bottom_weight[i * nyx + yx];
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int c = 0; c < nc; ++c) {
        bottom_data_vector.push_back(
            std::make_pair(bottom_data[i * nyxc + c * nyx + yx], c));
      }
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      // check if true label is in top k predictions
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == true_label) {
          accuracy_sum += weight;
          break;
        }
      }
    }
  }
  CHECK(!std::isnan(accuracy_sum));

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy_sum / ni;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(AccuracyMapLayer);
REGISTER_LAYER_CLASS(ACCURACY_MAP, AccuracyMapLayer);
}  // namespace caffe

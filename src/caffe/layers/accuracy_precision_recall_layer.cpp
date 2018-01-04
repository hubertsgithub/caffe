#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/accuracy_precision_recall_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AccuracyPrecisionRecallLayer<Dtype>::LayerSetUp( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top ) 
{
  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void AccuracyPrecisionRecallLayer<Dtype>::Reshape( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Number of labels must match number of predictions; ";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  for (int k = 0; k < 5; ++k)
    top[k]->Reshape(top_shape);
}

template <typename Dtype>
void AccuracyPrecisionRecallLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();

  double tp = 1e-5;
  double fp = 1e-5;
  double tn = 1e-5;
  double fn = 1e-5;
  for (int i = 0; i < bottom[0]->count(); i++)
  {
    const int label_value = static_cast<int>(bottom_label[i]);
    if (has_ignore_label_ && label_value == ignore_label_)
      continue;

    if (label_value == 1)
    {
      if (bottom_data[i] >= (Dtype)0.5)
        tp++;
      else
        fn++;
    }
    else
    {
      if (bottom_data[i] >= (Dtype)0.5)
        fp++;
      else
        tn++;
    }
  }

  double precision = tp / (tp + fp);
  double recall = tp / (tp + fn);
  double tpr = tp / (tp + fn);
  double fpr = tn / (tn + fp);

  top[0]->mutable_cpu_data()[0] = (Dtype)( (2.0 * precision * recall) / (precision + recall));
  top[1]->mutable_cpu_data()[0] = (Dtype)( precision );
  top[2]->mutable_cpu_data()[0] = (Dtype)( recall );
  top[3]->mutable_cpu_data()[0] = (Dtype)( tpr );
  top[4]->mutable_cpu_data()[0] = (Dtype)( fpr );
}

INSTANTIATE_CLASS(AccuracyPrecisionRecallLayer);
REGISTER_LAYER_CLASS(AccuracyPrecisionRecall);

}  // namespace caffe

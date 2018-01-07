#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/thr_accuracy_precision_recall_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ThrAccuracyPrecisionRecallLayer<Dtype>::LayerSetUp( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top ) 
{
  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void ThrAccuracyPrecisionRecallLayer<Dtype>::Reshape( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Number of labels must match number of predictions; ";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  for (int k = 0; k < 6; ++k)
    top[k]->Reshape(top_shape);
}

template <typename Dtype>
void ThrAccuracyPrecisionRecallLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();

  double best_thr = 0.0;
  double best_precision = 0.0;
  double best_recall = 0.0;
  double best_tnr = 0.0;
  double best_accuracy = 0.0;
  double best_fm = 0.0;

  for (double thr = 0.05; thr < 0.55; thr += 0.05)
  {
    double tp = 1e-5;
    double fp = 1e-5;
    double tn = 1e-5;
    double fn = 1e-5;
    for (int i = 0; i < bottom[0]->count(); ++i)
    {
      const int label_value = static_cast<int>(bottom_label[i]);
      if (has_ignore_label_ && label_value == ignore_label_)
        continue;

      if (label_value == 1)
      {
        if ((double)bottom_data[i] >= thr)
          tp++;
        else
          fn++;
      }
      else
      {
        if ((double)bottom_data[i] >= thr)
          fp++;
        else
          tn++;
      }
    }

    double precision = tp / (tp + fp);
    double recall = tp / (tp + fn);
    double tnr = tn / (tn + fp);
    double accuracy = (tp + tn) / (tp + tn + fp + fn);
    double fm = ((2.0 * precision * recall) / (precision + recall)); // F1 measure
    // double fm = (5.0 * tp) / (5.0 * tp + 4.0 * fn + fp);  // F2 measure

    // std::cout << "thr: " << thr << " fm: " << fm << " pr: " << precision << " rec: " << recall << " tnr: " << tnr << " acc: " << accuracy << std::endl;

    if (fm >= best_fm)
    {      
      best_thr = thr;
      best_precision = precision;
      best_recall = recall;
      best_tnr = tnr;
      best_accuracy = accuracy;
      best_fm = fm;

      // std::cout << "=> thr: " << best_thr << " fm: " << best_fm << " pr: " << best_precision << " rec: " << best_recall << " tnr: " << best_tnr << " acc: " << best_accuracy << std::endl;
    }
  }
  // std::cout << tp << " " << tn << " " << fp << " " << fn << std::endl;

  top[0]->mutable_cpu_data()[0] = (Dtype)best_fm;
  top[1]->mutable_cpu_data()[0] = (Dtype)best_precision;
  top[2]->mutable_cpu_data()[0] = (Dtype)best_recall;
  top[3]->mutable_cpu_data()[0] = (Dtype)best_tnr;
  top[4]->mutable_cpu_data()[0] = (Dtype)best_accuracy;
  top[5]->mutable_cpu_data()[0] = (Dtype)best_thr;
}

INSTANTIATE_CLASS(ThrAccuracyPrecisionRecallLayer);
REGISTER_LAYER_CLASS(ThrAccuracyPrecisionRecall);

}  // namespace caffe

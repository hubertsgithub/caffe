#ifndef CAFFE_THR_ACCURACY_PRECISION_RECALL_LAYER_HPP_
#define CAFFE_THR_ACCURACY_PRECISION_RECALL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes F-score / precision / recall / true negative rate / accuracy  for a multi-label classification task  / automatic prob threshold based on F-score
 */
template <typename Dtype>
class ThrAccuracyPrecisionRecallLayer : public Layer<Dtype> {
 public:
  explicit ThrAccuracyPrecisionRecallLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ThrAccuracyPrecisionRecall"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  virtual inline int MinTopBlobs() const { return 6; }
  virtual inline int MaxTopBlos() const { return 6; }

 protected:
  /**
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$ x @f$, a Blob with values in
   *      @f$ [-\infty, +\infty] @f$ indicating the predicted score for each of
   *      the @f$ K = CHW @f$ classes. Each @f$ x_n @f$ is mapped to a predicted
   *      label @f$ \hat{l}_n @f$ given by its maximal index:
   *      @f$ \hat{l}_n = \arg\max\limits_k x_{nk} @f$
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the labels @f$ l @f$, an integer-valued Blob with values
   *      @f$ l_n \in [0, 1] @f$
   *      indicating the correct 0/1 class label
   * @param top output Blob vector (length 7)
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      the geometric mean of precision/recall (F-score): @f$
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      precision: @f$
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      recall: @f$
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      true negative rate: @f$
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      accuracy: @f$
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      threshold that optimizes F-measure: @f$
   */

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- ThrAccuracyPrecisionRecallLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
};

}  // namespace caffe

#endif  // CAFFE_ACCURACY_LAYER_HPP_

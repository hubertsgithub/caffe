#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
MultiImageDataLayer<Dtype>::MultiImageDataLayer(const LayerParameter& param)
      : BasePrefetchingMultiDataLayer<Dtype>(param),
      	label_transform_param_(param.label_transform_param()),
      label_transformer_(label_transform_param_) {}

template <typename Dtype>
MultiImageDataLayer<Dtype>::~MultiImageDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void MultiImageDataLayer<Dtype>::LoadImageToSlot(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top, bool isTop, int index, const std::string& imgPath, const int new_height, const int new_width, const bool is_color) {
  Blob<Dtype>* blob;
  if (isTop) {
  	  blob = top[index];
  } else {
  	  blob = bottom[index];
  }

  // NOTE: The crop size should be the same for the labels and the data!
  int crop_size;
  Blob<Dtype>* prefetch_d;
  Blob<Dtype>* trafo_d;
  if (index == 0) {
  	  prefetch_d = &this->prefetch_data_;
  	  trafo_d = &this->transformed_data_;
 	  crop_size = this->layer_param_.transform_param().crop_size();
  } else {
  	  prefetch_d = this->prefetch_labels_[index-1].get();
  	  trafo_d = this->transformed_labels_[index-1].get();
 	  crop_size = this->layer_param_.label_transform_param().crop_size();
  }

  DLOG(INFO) << "blob: " << blob;
  DLOG(INFO) << "prefetch_d: " << prefetch_d;
  DLOG(INFO) << "trafo_d: " << trafo_d;

  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(imgPath, new_height, new_width, is_color);
  const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;
  // image
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  if (crop_size > 0) {
    blob->Reshape(batch_size, channels, crop_size, crop_size);
    prefetch_d->Reshape(batch_size, channels, crop_size, crop_size);
    trafo_d->Reshape(1, channels, crop_size, crop_size);
  } else {
    blob->Reshape(batch_size, channels, height, width);
    prefetch_d->Reshape(batch_size, channels, height, width);
    trafo_d->Reshape(1, channels, height, width);
  }
  LOG(INFO) << "output data size: " << blob->num() << ","
      << blob->channels() << "," << blob->height() << ","
      << blob->width();
}

template <typename Dtype>
int MultiImageDataLayer<Dtype>::ExactNumTopBlobs() const {
	int num = this->layer_param_.multi_prefetch_data_param().label_num()+1;
	DLOG(INFO) << "Number of top blobs: " << num;
	return num;
}

template <typename Dtype>
void MultiImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const int new_label_height = this->layer_param_.image_data_param().new_label_height();
  const int new_label_width = this->layer_param_.image_data_param().new_label_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  const bool label_is_color  = this->layer_param_.image_data_param().label_is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();
  // The BasePrefetchingMultiDataLayer took care of allocating the right number of label blobs

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  CHECK((new_label_height == 0 && new_label_width == 0) ||
      (new_label_height > 0 && new_label_width > 0)) << "Current implementation requires "
      "new_label_height and new_label_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  // The file and the label are both images
  string filename;
  string line;
  while (getline(infile, line)) {
  	stringstream ss(line);
  	ss >> filename;
  	LOG(INFO) << "Image data: " << filename;

    vector<string> labels;
  	for (int i = 0; i < this->prefetch_labels_.size(); ++i) {
  		string label;
  		ss >> label;
  		labels.push_back(label);
  		LOG(INFO) << "Image label: " << label;
  	}
    lines_.push_back(std::make_pair(filename, labels));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Set up transformed label array, it should have the same size as prefetch_labels
  this->transformed_labels_.resize(this->prefetch_labels_.size());
  for (int i = 0; i < this->prefetch_labels_.size(); ++i) {
  	  this->transformed_labels_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  }

  std::string imgPath = root_folder + lines_[lines_id_].first;
  this->LoadImageToSlot(bottom, top, true, 0, imgPath, new_height, new_width, is_color);
  // label images
  for (int i = 0; i < this->prefetch_labels_.size(); ++i) {
	  imgPath = root_folder + lines_[lines_id_].second[i];
	  this->LoadImageToSlot(bottom, top, true, 1+i, imgPath, new_label_height, new_label_width, label_is_color);
  }
}

template <typename Dtype>
void MultiImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void MultiImageDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  vector<Dtype*> top_labels;
  for (int i = 0; i < this->prefetch_labels_.size(); ++i) {
  	  top_labels.push_back(this->prefetch_labels_[i]->mutable_cpu_data());
  }

  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const int new_label_height = image_data_param.new_label_height();
  const int new_label_width = image_data_param.new_label_width();
  const bool is_color = image_data_param.is_color();
  const bool label_is_color  = image_data_param.label_is_color();
  string root_folder = image_data_param.root_folder();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
	string img_path = root_folder + lines_[lines_id_].first;
    DLOG(INFO) << "Loading image " << img_path << " as data";
    cv::Mat cv_img = ReadImageToCVMat(img_path,
                                    new_height, new_width, is_color);
    if (!cv_img.data) {
      DLOG(ERROR) << "Couldn't load image " << img_path;
      continue;
    }

    vector<cv::Mat> cv_img_labels;
    for (int i = 0; i < this->prefetch_labels_.size(); ++i) {
		string label_img_path = root_folder + lines_[lines_id_].second[i];
		DLOG(INFO) << "Loading image " << label_img_path << " as label #" << i;
		cv::Mat cv_img_label = ReadImageToCVMat(label_img_path,
                                    new_label_height, new_label_width, label_is_color);
		if (!cv_img_label.data) {
		  DLOG(ERROR) << "Couldn't load image " << label_img_path;
		  continue;
		}
    	cv_img_labels.push_back(cv_img_label);
	}

    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_.Transform(cv_img, &(this->transformed_data_));

    for (int i = 0; i < this->prefetch_labels_.size(); ++i) {
		offset = this->prefetch_labels_[i]->offset(item_id);
		this->transformed_labels_[i]->set_cpu_data(top_labels[i] + offset);
		this->label_transformer_.Transform(cv_img_labels[i], &(*this->transformed_labels_[i]));
	}
    trans_time += timer.MicroSeconds();

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(MultiImageDataLayer);
REGISTER_LAYER_CLASS(MULTI_IMAGE_DATA, MultiImageDataLayer);
}  // namespace caffe

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <sstream>
#include <string>
#include <algorithm>
#include <utility>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/stat.h>
#include <boost/random/uniform_real.hpp>
#include <boost/format.hpp>
#include <omp.h>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

using std::min;
using std::max;

namespace caffe {

volatile double read_counter(void) {
  struct timeval tv;
  double t;

  gettimeofday(&tv,NULL);
  t=(double)tv.tv_sec+(double)tv.tv_usec*1e-6;
  return(t);
}

static bool posixpath_exists(const std::string &s) {
  struct stat st;
  return(stat(s.c_str(),&st)==0);
}

template <typename Dtype>
cv::Mat footprint_cvmat(cv::Mat source, Dtype ax, Dtype ay, Dtype bx, Dtype by, cv::Scalar border_value, int new_width, int new_height) {
  // Input: H X W X C source image and normalized footprint a, b in
  //   the coordinate frame of the input.
  // Goal: create a new image which is the size of the footprint and
  //   any pixels which fall inside the input take their value from
  //   input otherwise the value is the border value.
  // opencv is band-interleaved (e.g., BGRBGRBGR).

  CHECK(source.data);
  const int w=source.cols;
  const int h=source.rows;
  const int c=source.channels();
  const int x0=static_cast<int>(ax*w+0.5);
  const int y0=static_cast<int>(ay*h+0.5);
  const int x1=static_cast<int>(bx*w+0.5);
  const int y1=static_cast<int>(by*h+0.5);
  CHECK_GT(x1,x0);
  CHECK_GT(y1,y0);
  const int dwidth=x1-x0;
  const int dheight=y1-y0;
  //cv::Mat result(dheight,dwidth,(sizeof(Dtype)==8 ? CV_64FC(c) : CV_32FC(c)),border_value);
  cv::Mat result(dheight,dwidth,CV_8UC(c),border_value);
  for(int j=max(0,y0);j<y1;j++) {
    if(j>=h) break;
    // j in [max(0,y0), min(h-1,y1)]
    // i in [max(0,x0), min(w-1,x1)]
    CHECK(j>=max(0,y0));
    CHECK(j<=min(h-1,y1));
    CHECK(j-y0>=0);
    CHECK(j-y0<=dheight-1);
    CHECK(max(0,x0)-x0>=0);
    CHECK(max(0,x0)-x0<=dwidth-1);
    uchar* src=source.ptr<uchar>(j)+max(0,x0)*c;
    uchar* dest=result.ptr<uchar>(j-y0)+(max(0,x0)-x0)*c;
    memcpy(dest,src,(min(w-1,x1)-max(0,x0))*c);
  }
  //cv::Mat cv_img;
  //if (new_height > 0 && new_width > 0) {
    //cv::resize(result, cv_img, cv::Size(new_width, new_height));
  //} else {
    //cv_img = result;
  //}
  //return cv_img;
  if (new_height > 0 && new_width > 0) {
    if (result.rows == new_height && result.cols == new_width) {
      return result;
    } else if (result.rows > new_height || result.cols > new_width) {
      cv::Mat cv_img;
      cv::resize(result, cv_img, cv::Size(new_width, new_height),0,0,cv::INTER_AREA);
      return cv_img;
    } else {
      cv::Mat cv_img;
      cv::resize(result, cv_img, cv::Size(new_width, new_height));
      return cv_img;
    }
  } else {
    return result;
  }
  LOG(FATAL) << "Your assumptions about the universe are incorrect.";
}

template <typename Dtype>
cv::Mat ReadImageFootprintToCVMat(const string& path, const int height, const int width, const bool is_color, Dtype ax, Dtype ay, Dtype bx, Dtype by) {
  //LOG(INFO) << path << " " << ax << " " << ay << " " << bx << " " << by;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(path, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(FATAL) << "Could not open or find file " << path;
    return cv_img_origin;
  }
  if(ax>0 || ay>0 || bx<1 || by<1) {
    const int w=cv_img_origin.cols;
    const int h=cv_img_origin.rows;
    int x0=static_cast<int>(ax*w+0.5);
    int y0=static_cast<int>(ay*h+0.5);
    int x1=static_cast<int>(bx*w+0.5);
    int y1=static_cast<int>(by*h+0.5);
    x1=(x1<x0+1 ? x0+1 : x1); // Note: this logic may cause us to read outside the
    y1=(y1<y0+1 ? y0+1 : y1); // footprint. Do not use small footprints!
    cv_img_origin=cv_img_origin(cv::Rect(x0,y0,x1-x0,y1-y0));
  }
  if (height > 0 && width > 0) {
    if (cv_img_origin.rows == height && cv_img_origin.cols == width) {
      return cv_img_origin;
    } else if (cv_img_origin.rows > height || cv_img_origin.cols > width) {
      cv::Mat cv_img;
      cv::resize(cv_img_origin, cv_img, cv::Size(width, height),0,0,cv::INTER_AREA);
      return cv_img;
    } else {
      cv::Mat cv_img;
      cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
      return cv_img;
    }
  } else {
    return cv_img_origin;
  }
  LOG(FATAL) << "Your assumptions about the universe are incorrect.";
}

template <typename Dtype>
MultiImageDataLayer<Dtype>::~MultiImageDataLayer<Dtype>() {
  CPUTimer join_timer;
  join_timer.Start();
  this->JoinPrefetchThread();
  LOG(INFO) << "Join time: " << join_timer.MilliSeconds() << " ms.";
}

template <typename Dtype>
void MultiImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.multi_image_data_param().new_height();
  const int new_width = this->layer_param_.multi_image_data_param().new_width();
  const int num_image = this->layer_param_.multi_image_data_param().num_image();
  const int num_label = this->layer_param_.multi_image_data_param().num_label();
  const bool oversample = this->layer_param_.multi_image_data_param().oversample();
  const bool verbose = this->layer_param_.multi_image_data_param().verbose();
  const int sample_maximum = this->layer_param_.multi_image_data_param().sample_maximum();
  const bool check_existence = this->layer_param_.multi_image_data_param().check_existence();
  const bool require_existence = this->layer_param_.multi_image_data_param().require_existence();
  const bool edge_fill = this->layer_param_.multi_image_data_param().edge_fill();
  const float ratio_jitter = this->layer_param_.multi_image_data_param().ratio_jitter();
  const float position_jitter = this->layer_param_.multi_image_data_param().position_jitter();
  const float spatial_scale_jitter = this->layer_param_.multi_image_data_param().spatial_scale_jitter();
  const bool unbalanced = this->layer_param_.multi_image_data_param().unbalanced();
  string root_folder = this->layer_param_.multi_image_data_param().root_folder();

  if(oversample) {
    CHECK(ratio_jitter==0);
    CHECK(position_jitter==0);
    CHECK(spatial_scale_jitter==0);
  }
  CHECK_GE(ratio_jitter, 0.0) << "ratio_jitter must be >= 0.";
  CHECK_GE(position_jitter, 0.0) << "position_jitter must be in the range [0, 0.5]";
  CHECK_LE(position_jitter, 0.5) << "position_jitter must be in the range [0, 0.5]";
  CHECK_GE(spatial_scale_jitter, 0.0) << "spatial_scale must be in the range [0, 1]";
  CHECK_LE(spatial_scale_jitter, 1.0) << "spatial_scale must be in the range [0, 1]";
  if(edge_fill) {
    CHECK(this->layer_param_.multi_image_data_param().border_value_size());
    if(num_image>1) {
      NOT_IMPLEMENTED; // Actually, it is implemented, but not tested
    }
  }

  // prepare openmp
  omp_set_num_threads(8);

  // label_index_set_ is a list of line indices organized by label.
  CHECK_GT(num_label,0);
  for(int i=0;i<num_label;i++) {
    vector<int> v;
    label_index_set_.push_back(v);
  }

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.multi_image_data_param().source();
  LOG(INFO) << "Opening source file " << source;
  {
    // Source file record format:
    //   label s path,a,b,c ...
    // label is either an integer or a comma-separated list
    //   label in {0, ..., num_label-1} is the integer label
    //   l1,...,lm in {0, ..., num_label-1}^m are multi-labels
    // There are various parameters for sample augmentation
    //   s in (0, 1] is the scale jitter
    //   r >=0 is the aspect ratio jitter (from prototxt)
    //   j in [0,0.5) is the position jitter (from prototxt)
    // There are num_image images each described by a comma-separated list
    //   a,b in [0, 1] denote the footprint
    //   c in {1, 3} is the number of channels
    std::ifstream infile(source.c_str());
    int index=0;
    int ignored=0;
    int line_number=0;
    string labeldesc;
    while (line_number<sample_maximum && (infile >> labeldesc)) {
      struct line l;
      if(labeldesc.find(",")!=string::npos) {
        NOT_IMPLEMENTED;
      }
      else {
        l.label=atoi(labeldesc.c_str());
      }
      //LOG(INFO) << "label:  " << l.label;
      CHECK(infile >> l.s);
      l.r=ratio_jitter;
      l.j=position_jitter;
      CHECK_GT(l.s, 0) << "Scale must be in the range (0, 1]";
      CHECK_LE(l.s, 1) << "Scale must be in the range (0, 1]";
      //LOG(INFO) << "jitter: " << l.s << " " << l.r << " " << l.j;
      int incomplete=0;
      string imagedesc;
      for(int z=0;z<num_image;z++) {
        CHECK(infile >> imagedesc);
        replace(imagedesc.begin(),imagedesc.end(),',',' ');
        struct patch_info i;
        std::istringstream ss(imagedesc);
        ss >> i.path
           >> i.ax >> i.ay >> i.bx >> i.by
           >> i.c;
        if(check_existence) {
          bool e=posixpath_exists(root_folder+i.path);
          if(!e) { incomplete=1; }
          if(!e && require_existence) {
            LOG(FATAL) << "Missing an image file (" << (root_folder+i.path) << ")!";
            exit(1);
          }
        }
        l.image.push_back(i);
        //LOG(INFO) << "image:  " << i.path << " " << i.ax << " " << i.ay << " " << i.bx << " " << i.by << " " << i.c;
      }
      if(!incomplete) {
        CHECK_LT(l.label,num_label) << "Label must not exceed num_label.";
        lines_.push_back(l);
        label_index_set_[l.label].push_back(index);
        unbalanced_index_set_.push_back(index);
        index=index+1;
      }
      else {
        ignored=ignored+1;
      }
      line_number+=1;
    }
    LOG(INFO) << "Found " << lines_.size() << " samples.";
    LOG(INFO) << "Ignored " << ignored << " incomplete samples.";
    CHECK_GT(lines_.size(),0) << "There must be at least one valid sample.";
    CHECK(lines_.size()==unbalanced_index_set_.size());
  }
  if(verbose) {
    // number of samples per category
    for(int i=0;i<num_label;i++) {
      LOG(INFO) << "  " << i << ": " << label_index_set_[i].size();
    }
  }
  // Future note: to support database backends, such as LMDB, all we need
  // to do is check each image in the db and push the index onto label_index_set_.
  // The db must support random access.

  // randomly shuffle data
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  if (this->layer_param_.multi_image_data_param().shuffle()) {
    LOG(INFO) << "Shuffling samples.";
    ShuffleImages();
  }

  // Initialize tail and randomly skip a few data points (if requested)
  {
    if(unbalanced) {
      unbalanced_index_set_tail_=0;
    }
    else {
      for(int i=0;i<num_label;i++) {
        label_index_set_tail_.push_back(0);
      }
    }
    if (this->layer_param_.multi_image_data_param().rand_skip()) {
      // This is obsolete. Instead, call Caffe::set_initial_skip(n)
      // before creating the Net.
      Skip(caffe_rng_rand() % this->layer_param_.multi_image_data_param().rand_skip());
    }
  }

  // Read an image, and use it to initialize the top blob shape.
  int channels=0;
  int height=0;
  int width=0;
  //struct line &l=lines_[label_index_set_[0][label_index_set_tail_[0]]]; // the correct way to read a sample
  struct line &l=lines_[0]; // but, we just need an arbitrary sample
  for(int i=0;i<num_image;i++) {
    struct patch_info &im=l.image[i];
    if(!edge_fill) {
      CHECK_GE(im.ax,0); CHECK_LE(im.bx,1);
      CHECK_GE(im.ay,0); CHECK_LE(im.by,1);
    }
    cv::Mat cv_img;
    if(edge_fill) {
      cv::Mat I = ReadImageToCVMat(root_folder + im.path, im.c==3);
      cv_img = footprint_cvmat(I, im.ax, im.ay, im.bx, im.by, (im.c==3 ? cv::Scalar(0,0,0) : cv::Scalar(0)), new_width, new_height);
    }
    else {
      cv_img = ReadImageFootprintToCVMat(root_folder + im.path, new_height, new_width, im.c==3, im.ax, im.ay, im.bx, im.by);
    }
    channels+=cv_img.channels();
    if(i==0) {
      height=cv_img.rows;
      width=cv_img.cols;
    }
    else {
      CHECK_EQ(height, cv_img.rows);
      CHECK_EQ(width, cv_img.cols);
    }
  }
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.multi_image_data_param().batch_size();
  if(oversample) {
    CHECK_GT(crop_size,0) << "Must specify data transformer crop_size when oversampling.";
    //CHECK((batch_size % 10)==0) << "batch_size must be a multiple of 10 when oversampling.";
    CHECK((batch_size % 15)==0) << "batch_size must be a multiple of 15 when oversampling.";
    CHECK(this->layer_param_.transform_param().mirror()==false) << "Must not specify data transformer mirror when oversampling.";
  }
  if (crop_size > 0) {
    top[0]->Reshape(batch_size, channels, crop_size, crop_size);
    this->prefetch_blob_[0].Reshape(batch_size, channels, crop_size, crop_size);
    this->transformed_data_.Reshape(1, channels, crop_size, crop_size);
  } else {
    top[0]->Reshape(batch_size, channels, height, width);
    this->prefetch_blob_[0].Reshape(batch_size, channels, height, width);
    this->transformed_data_.Reshape(1, channels, height, width);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // Initialize label shape.
  top[1]->Reshape(batch_size, 1, 1, 1);
  this->prefetch_blob_[1].Reshape(batch_size, 1, 1, 1);
  LOG(INFO) << "output label size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << ","
      << top[1]->width();
}

template <typename Dtype>
void MultiImageDataLayer<Dtype>::Skip(int n) {
  const bool unbalanced = this->layer_param_.multi_image_data_param().unbalanced();

  if(n) {
    LOG(INFO) << "Skipping " << n << " data points.";
  }
  if(unbalanced) {
    CHECK_GT(lines_.size(), unbalanced_index_set_tail_+n) << "Not enough points to skip";
    unbalanced_index_set_tail_ += n;
  }
  else {
    NOT_IMPLEMENTED;
  }
}

template <typename Dtype>
void MultiImageDataLayer<Dtype>::ShuffleImages() {
  double t0=read_counter();
  const int num_label = this->layer_param_.multi_image_data_param().num_label();
  const int verbose = this->layer_param_.multi_image_data_param().verbose();
  const bool unbalanced = this->layer_param_.multi_image_data_param().unbalanced();
  caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  if(unbalanced) {
    shuffle(unbalanced_index_set_.begin(), unbalanced_index_set_.end(), prefetch_rng);
  }
  else {
    for(int i=0;i<num_label;i++) {
      if(label_index_set_[i].size()>0) {
        shuffle(label_index_set_[i].begin(), label_index_set_[i].end(), prefetch_rng);
      }
    }
  }
  double t1=read_counter();
  if(verbose) { LOG(INFO) << "List shuffled in " << (t1-t0) << " seconds."; }
}

template <typename Dtype>
cv::Mat MultiImageDataLayer<Dtype>::ToGrayscale(const cv::Mat& cv_img) {
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  cv::Mat cv_img_gray = cv_img;
  for (int h = 0; h < img_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    uchar* ptr_gray = cv_img_gray.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < img_width; ++w) {
      Dtype gray_pixel = 0;
      for (int c = 0; c < img_channels; ++c) {
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        gray_pixel += pixel;
      }
      gray_pixel /= img_channels;
      img_index -= img_channels;
      for (int c = 0; c < img_channels; ++c) {
        // int top_index = (c * height + h) * width + w;
        ptr_gray[img_index++] = (uchar)gray_pixel;
      }
    }
  }

  return cv_img_gray;
}


// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void MultiImageDataLayer<Dtype>::InternalThreadEntry() {
  double t0=read_counter();
  CHECK(this->prefetch_blob_[0].count());
  CHECK(this->transformed_data_.count());
  Dtype* top_data = this->prefetch_blob_[0].mutable_cpu_data();
  Dtype* top_label = this->prefetch_blob_[1].mutable_cpu_data();
  MultiImageDataParameter multi_image_data_param = this->layer_param_.multi_image_data_param();
  const int batch_size = multi_image_data_param.batch_size();
  const int new_height = multi_image_data_param.new_height();
  const int new_width = multi_image_data_param.new_width();
  const int num_image = multi_image_data_param.num_image();
  const int num_label = multi_image_data_param.num_label();
  const int oversample = multi_image_data_param.oversample();
  string root_folder = multi_image_data_param.root_folder();
  const bool verbose = multi_image_data_param.verbose();
  const bool unbalanced = multi_image_data_param.unbalanced();
  const bool edge_fill = multi_image_data_param.edge_fill();
  const bool grayscale = multi_image_data_param.grayscale();
  const float spatial_scale_jitter = this->layer_param_.multi_image_data_param().spatial_scale_jitter();
  caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());

  static int static_batch_count = 0;
  if (verbose) {
    LOG(INFO) << "Starting batch " << static_batch_count << "...";
  }

  vector<int> S;
  if(unbalanced) {
    while(S.size()<batch_size) {
      if(oversample) {
        //for(int j=0;j<10;j++) S.push_back(unbalanced_index_set_[unbalanced_index_set_tail_]);
        for(int j=0;j<15;j++) S.push_back(unbalanced_index_set_[unbalanced_index_set_tail_]);
      }
      else {
        S.push_back(unbalanced_index_set_[unbalanced_index_set_tail_]);
      }
      unbalanced_index_set_tail_++;
      if(unbalanced_index_set_tail_>=unbalanced_index_set_.size()) {
        unbalanced_index_set_tail_=0;
        if(multi_image_data_param.shuffle()) { ShuffleImages(); }
      }
    }
  }
  else {
    // L, a randomly permuted set of labels
    vector<int> L;
    for(int i=0;i<num_label;i++) L.push_back(i);
    shuffle(L.begin(), L.end(), prefetch_rng);

    // Goal: build a set S of samples which is balanced in each category
    // Algorithm:
    //   repeat
    //     l <- next label from randomly permuted set of labels
    //     ignore l if there are no samples in that category
    //     else
    //       idx <- next index from index_set[l]
    //       reset index_set if it is exhausted
    //       append idx to S
    while(S.size()<batch_size) {
      for(int i=0;i<num_label && S.size()<batch_size;i++) {
        int l=L[i];
        vector<int> &index_set=label_index_set_[l];
        if(index_set.size()<1) continue;
        int tail=label_index_set_tail_[l];
        int idx=index_set[tail];
        if(tail+1>=index_set.size()) {
          if (verbose) { LOG(INFO) << "Label " << l << " exhausted."; }
          if(multi_image_data_param.shuffle()) {
            shuffle(index_set.begin(), index_set.end(), prefetch_rng);
            if (verbose) { LOG(INFO) << "Label " << l << " shuffled."; }
          }
          label_index_set_tail_[l]=0;
        }
        else {
          label_index_set_tail_[l]=tail+1;
        }
        if(oversample) {
          //for(int j=0;j<10;j++) S.push_back(idx);
          for(int j=0;j<15;j++) S.push_back(idx);
        }
        else {
          S.push_back(idx);
        }
      }
    }
  }
  CHECK_EQ(S.size(),batch_size);

  // The batch is lines_[S[0]], lines_[S[1]], ...

  // ================================================================
  // Parallel batch filling loop starts here
  // ================================================================
  #pragma omp parallel for schedule(dynamic, 1)
  for(int j=0;j<batch_size;j++) {
    //if(oversample && (j % 10)) {
    //  // When oversampling only slots 0, 10, 20, ... do any processing
    //  continue;
    //}
    if(oversample && (j % 5)) {
      // When oversampling only slots 0, 5, 10, ... do any processing
      continue;
    }

    struct line &l=lines_[S[j]];

    // ================================================================
    // First: Calculate footprint jitter
    // ================================================================
    Dtype s, sx, sy, jx, jy;

    // critical block to protect shared state: prefetch_rng
    #pragma omp critical(multi_data_layer_rng)
    {
      // random scale
      s = 1;
      if (!oversample && l.s < 1) {
        // geometric mean of l.s and 1
        const Dtype mid_scale = sqrt(l.s * 1.);
        if (spatial_scale_jitter > 0) {
          // spatial_scale_jitter interpolates between not doing scale jitter (0) and doing scale jitter (1)
          const Dtype min_scale = max<Dtype>(l.s, mid_scale + spatial_scale_jitter * (l.s - mid_scale));
          const Dtype max_scale = min<Dtype>(1.0, mid_scale + spatial_scale_jitter * (1.0 - mid_scale));
          boost::uniform_real<> distribution(min_scale, max_scale);
          s = distribution(*prefetch_rng);
        } else {
          s = mid_scale;
        }
      }
      if (oversample) {
        const Dtype mid_scale = sqrt(l.s * 1.);
        switch ((j / 5) % 3) {
          case 0: s = l.s; break;
          case 1: s = mid_scale; break;
          case 2: s = 1.0; break;
        }
      }
      // random aspect ratio constrained by maximum scale
      sx = s;
      sy = s;
      if (l.r > 0) {
        const Dtype min_scale = s / (1 + l.r);
        const Dtype max_scale = min<Dtype>(1.0, s * (1 + l.r));
        boost::uniform_real<> distribution(min_scale, max_scale);
        sx = distribution(*prefetch_rng);
        sy = distribution(*prefetch_rng);
      }
      // random position jitter constrained by selected scale
      jx = min<Dtype>(l.j, (1 - sx) * 0.5);
      jy = min<Dtype>(l.j, (1 - sy) * 0.5);
      if (jx > 0) {
        boost::uniform_real<> distribution(-jx, jx);
        jx = distribution(*prefetch_rng);
      }
      if (jy > 0) {
        boost::uniform_real<> distribution(-jy, jy);
        jy = distribution(*prefetch_rng);
      }
    } // end critical

    // ================================================================
    // Next: read the input images (maybe more than one)
    // ================================================================
    vector<cv::Mat> mv;
    int mv_channel_offset=0;
    cv::Mat cv_img[num_image];
    for(int i=0;i<num_image;i++) {
      struct patch_info &im=l.image[i];
      // get a blob
      // a <- (a+b)/2+j(b-a)-s(b-a)/2
      // b <- (a+b)/2+j(b-a)+s(b-a)/2
      Dtype ax = (0.5 - jx + sx * 0.5) * im.ax + (0.5 + jx - sx * 0.5) * im.bx;
      Dtype ay = (0.5 - jy + sy * 0.5) * im.ay + (0.5 + jy - sy * 0.5) * im.by;
      Dtype bx = (0.5 - jx - sx * 0.5) * im.ax + (0.5 + jx + sx * 0.5) * im.bx;
      Dtype by = (0.5 - jy - sy * 0.5) * im.ay + (0.5 + jy + sy * 0.5) * im.by;

      if(edge_fill) {
        cv::Mat I = ReadImageToCVMat(root_folder + im.path, im.c==3);
        // copy footprint from input (I) and pad any missing pixels with border_value
        cv_img[i] = footprint_cvmat(
          I, ax, ay, bx, by, (
           (im.c == 3) ?
             cv::Scalar(
               multi_image_data_param.border_value(mv_channel_offset+0),
               multi_image_data_param.border_value(mv_channel_offset+1),
               multi_image_data_param.border_value(mv_channel_offset+2))
           :
             cv::Scalar(multi_image_data_param.border_value(mv_channel_offset+0))
          ), new_width, new_height);
      }
      else {
        // clamp in case of rounding errors
        ax = (ax > 0 ? ax : 0);
        ay = (ay > 0 ? ay : 0);
        bx = (bx < 1 ? bx : 1);
        by = (by < 1 ? by : 1);
        cv_img[i] = ReadImageFootprintToCVMat(root_folder + im.path, new_height, new_width, im.c==3, ax, ay, bx, by);
      }
      CHECK(cv_img[i].data); // This is bad, but nothing to do about it now
      if (grayscale) {
        cv_img[i] = this->ToGrayscale(cv_img[i]);
      }
      mv.push_back(cv_img[i]);
      mv_channel_offset+=cv_img[i].channels();
    }

    //#pragma omp critical(multi_debug_image)
    //{
    //  for(int i=0;i<num_image;i++) { // debug
    //    static int index;
    //    cv::imwrite(str(boost::format("zzz%09d.png") % index),cv_img[i]);
    //    index+=1;
    //  }
    //}

    // ================================================================
    // Next: merge input images and pass to data transformer
    // ================================================================
    cv::Mat cv_blob;
    cv::merge(mv,cv_blob);

    // critical block to protect shared state: data_transformer_, transformed_data_
    #pragma omp critical(multi_data_layer_transform)
    {
      if(oversample) {
        //for(int i=0;i<10;i++) {
        for(int i=0;i<5;i++) {
          int offset = this->prefetch_blob_[0].offset(j+i);
          this->transformed_data_.set_cpu_data(top_data + offset);
          // Apply transformations (mirror, crop...) to the image
          this->data_transformer_->Transform(cv_blob, &(this->transformed_data_), i+1);
        }
      }
      else {
        int offset = this->prefetch_blob_[0].offset(j);
        this->transformed_data_.set_cpu_data(top_data + offset);
        // Apply transformations (mirror, crop...) to the image
        this->data_transformer_->Transform(cv_blob, &(this->transformed_data_));
      }
    }

    // ================================================================
    // Last: set label
    // ================================================================
    if(oversample) {
      //for(int i=0;i<10;i++) top_label[j+i] = l.label;
      for(int i=0;i<5;i++) top_label[j+i] = l.label;
    }
    else {
      top_label[j] = l.label;
    }
  }

  double t1=read_counter();
  if(verbose) {
    LOG(INFO) << "Batch " << static_batch_count << " prepared in " << (t1-t0) << " seconds.";
    static_batch_count ++;
  }
}

INSTANTIATE_CLASS(MultiImageDataLayer);
REGISTER_LAYER_CLASS(MultiImageData);

}  // namespace caffe

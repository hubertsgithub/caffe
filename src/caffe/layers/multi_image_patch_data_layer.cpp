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
MultiImagePatchDataLayer<Dtype>::MultiImagePatchDataLayer(const LayerParameter& param)
	: BasePrefetchingMultiDataLayer<Dtype>(param) {
}

template <typename Dtype>
MultiImagePatchDataLayer<Dtype>::~MultiImagePatchDataLayer<Dtype>() {
	this->JoinPrefetchThread();
}

template <typename Dtype>
void MultiImagePatchDataLayer<Dtype>::LoadImageToSlot(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top, bool isTop, int index, const std::string& imgPath, const int new_height, const int new_width, const bool is_color,
        const bool crop_first, const int crop_size) {
	// TODO: Move this function to a separate class and inherit MultiImageDataLayer + MultiImagePatchDataLayer from it!
	Blob<Dtype>* blob;
	if (isTop) {
		blob = top[index];
	} else {
		blob = bottom[index];
	}

	Blob<Dtype>* prefetch_d;
	Blob<Dtype>* trafo_d;
	prefetch_d = this->prefetch_data_[index].get();
	trafo_d = this->transformed_data_[index].get();

	DLOG(INFO) << "blob: " << blob;
	DLOG(INFO) << "prefetch_d: " << prefetch_d;
	DLOG(INFO) << "trafo_d: " << trafo_d;

	// Read an image, and use it to initialize the top blob.
	cv::Mat cv_img = ReadImageToCVMat(imgPath, 0, 0, is_color);
	const int img_height = cv_img.rows;
	const int img_width = cv_img.cols;

	int channels;
	if (is_color) {
		channels = 3;
	} else {
		channels = 1;
	}

	int height;
	if (crop_first) {
		if (new_height > 0) {
			// If crop first then resize later, the resized size will be final
			height = new_height;
		} else if (crop_size > 0) {
			// If we crop first but don't resize, the crop size will be final
			height = crop_size;
		} else {
			// If we don't crop or resize, the original resolution stays
			height = img_height;
		}
	} else {
		if (crop_size > 0) {
			// If we resize first then crop later, the crop size will be final
			height = crop_size;
		} else if (new_height > 0) {
			// If we resize first don't crop, the resize size will be final
			height = new_height;
		} else {
			// If we don't crop or resize, the original resolution stays
			height = img_height;
		}
	}

	int width;
	if (crop_first) {
		if (new_width > 0) {
			// If crop first then resize later, the resized size will be final
			width = new_width;
		} else if (crop_size > 0) {
			// If we crop first but don't resize, the crop size will be final
			width = crop_size;
		} else {
			// If we don't crop or resize, the original resolution stays
			width = img_width;
		}
	} else {
		if (crop_size > 0) {
			// If we resize first then crop later, the crop size will be final
			width = crop_size;
		} else if (new_width > 0) {
			// If we resize first don't crop, the resize size will be final
			width = new_width;
		} else {
			// If we don't crop or resize, the original resolution stays
			width = img_width;
		}
	}

	// image
	const int batch_size = this->layer_param_.image_data_param().batch_size();
	blob->Reshape(batch_size, channels, height, width);
	prefetch_d->Reshape(batch_size, channels, height, width);
	trafo_d->Reshape(1, channels, height, width);

	LOG(INFO) << "output data size: " << blob->num() << ","
	          << blob->channels() << "," << blob->height() << ","
	          << blob->width();
}

template <typename Dtype>
void MultiImagePatchDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
	string root_folder = this->layer_param_.image_data_param().root_folder();
	// The BasePrefetchingMultiDataLayer took care of allocating the right number of data blobs
	CHECK_EQ(top.size(), this->prefetch_data_.size()) << "The top count should match the prefetch data count.";

	// Read the file with filenames and labels
	const string& source = this->layer_param_.image_data_param().source();
	LOG(INFO) << "Opening file " << source;
	std::ifstream infile(source.c_str());
	///////////
	// A file looks like this: imagepath0 imagepath1 [...] imagepathn sim(0 or 1) patch0x patch0y patch1x patch1y [...] patchmx patchmy
	///////////
	MultiPrefetchDataParameter mpdp = this->layer_param_.multi_prefetch_data_param();
	int input_data_count = mpdp.input_data_count();
	const int image_count = input_data_count;
	const int patch_count = this->layer_param_.multi_image_patch_data_param().patch_count();
	CHECK_EQ(mpdp.data_transformations_size(), image_count) << "The data transformation count should match the number of input images.";
	CHECK_EQ(this->prefetch_data_.size(), image_count * patch_count + 1) << "The prefetch data size should match the number of input images times the number of patches + 1 (label).";

	string line;
	while (getline(infile, line)) {
		stringstream ss(line);

		vector<string> data_img_names;
		for (int i = 0; i < image_count + 1 + 2 * patch_count; ++i) {
			string img_name;
			ss >> img_name;
			CHECK(img_name.size() > 0) << "The number of elements in a row of the data file should be equal to image_count + 1 + 2 * patch_count";
			data_img_names.push_back(img_name);
			if (i < image_count) {
				DLOG(INFO) << "Image data #" << i << ": " << img_name;
			} else if (i == image_count) {
				DLOG(INFO) << "Sim: " << img_name;
			} else {
				DLOG(INFO) << "Patch data #" << (i - image_count - 1) / 2 << ((i - image_count - 1) % 2 == 0 ? "w" : "h") << ": " << img_name;
			}

		}
		lines_.push_back(data_img_names);
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

	// Allocate memory for all tops
	for (int iPatch = 0; iPatch < patch_count; ++iPatch) {
		for (int iImage = 0; iImage < image_count; ++iImage) {
			int i = iPatch * image_count + iImage;
			// Order of top blobs should be ij, i = 1 -> m (patch count), j = 1 -> n (input image count)
			// So if m = 2, n = 3
			// 11, 12, 13, 21, 22, 23
			// Same transformations for all patches! (except for the center of the crop of course)
			const int new_height = this->layer_param_.multi_prefetch_data_param().data_transformations(iImage).new_height();
			const int new_width  = this->layer_param_.multi_prefetch_data_param().data_transformations(iImage).new_width();
			const bool is_color  = this->layer_param_.multi_prefetch_data_param().data_transformations(iImage).is_color();
			const bool crop_first  = this->layer_param_.multi_prefetch_data_param().data_transformations(iImage).crop_first();
			const int crop_size = this->layer_param_.multi_prefetch_data_param().data_transformations(iImage).crop_size();
			CHECK((new_height == 0 && new_width == 0) ||
			      (new_height > 0 && new_width > 0)) << "Current implementation requires "
			              "new_height and new_width to be set at the same time.";

			string imgPath = root_folder + lines_[lines_id_][iImage];
			this->LoadImageToSlot(bottom, top, true, i, imgPath, new_height, new_width, is_color, crop_first, crop_size);
		}
	}
	// The last top contains the label
	const int batch_size = this->layer_param_.image_data_param().batch_size();
	int index = patch_count * image_count;
	Blob<Dtype>* blob;
	Blob<Dtype>* prefetch_d;
	Blob<Dtype>* trafo_d;
	blob = top[index];
	prefetch_d = this->prefetch_data_[index].get();
	trafo_d = this->transformed_data_[index].get();

	DLOG(INFO) << "blob: " << blob;
	DLOG(INFO) << "prefetch_d: " << prefetch_d;
	DLOG(INFO) << "trafo_d: " << trafo_d;
	blob->Reshape(batch_size, 1, 1, 1);
	prefetch_d->Reshape(batch_size, 1, 1, 1);
	trafo_d->Reshape(1, 1, 1, 1);

	LOG(INFO) << "output data size: " << blob->num() << ","
	          << blob->channels() << "," << blob->height() << ","
	          << blob->width();
}

template <typename Dtype>
void MultiImagePatchDataLayer<Dtype>::ShuffleImages() {
	caffe::rng_t* prefetch_rng =
	    static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void MultiImagePatchDataLayer<Dtype>::InternalThreadEntry() {
	DLOG(INFO) << "Prefetch thread started";

	CPUTimer batch_timer;
	batch_timer.Start();
	double read_time = 0;
	double trans_time = 0;
	CPUTimer timer;
	//  Check all blobs, we should have reshaped them by now
	CHECK_EQ(this->prefetch_data_.size(), this->transformed_data_.size()) << "The prefetch data size should match the transformed data size";
	for (int i = 0; i < this->prefetch_data_.size(); ++i) {
		CHECK(this->prefetch_data_[i]->count());
		CHECK(this->transformed_data_[i]->count());
	}

	vector<Dtype*> top_data;
	for (int i = 0; i < this->prefetch_data_.size(); ++i) {
		top_data.push_back(this->prefetch_data_[i]->mutable_cpu_data());
	}

	ImageDataParameter image_data_param = this->layer_param_.image_data_param();
	const int batch_size = image_data_param.batch_size();
	string root_folder = image_data_param.root_folder();
	const bool share_random_trafos = this->layer_param_.multi_prefetch_data_param().share_random_trafos();
	MultiPrefetchDataParameter mpdp = this->layer_param_.multi_prefetch_data_param();
	int input_data_count = mpdp.input_data_count();
	const int image_count = input_data_count;
	const int patch_count = this->layer_param_.multi_image_patch_data_param().patch_count();
	CHECK_EQ(mpdp.data_transformations_size(), image_count) << "The data transformation count should match the number of input images.";
	CHECK_EQ(this->prefetch_data_.size(), image_count * patch_count + 1) << "The prefetch data size should match the number of input images times the number of patches + 1 (label).";

	// datum scales
	const int lines_size = lines_.size();
	for (int item_id = 0; item_id < batch_size; ++item_id) {
		// get a blob
		timer.Start();
		CHECK_GT(lines_size, lines_id_);

		vector<cv::Mat> cv_imgs;
		for (int i = 0; i < image_count; ++i) {
			const int new_height = mpdp.data_transformations(i).new_height();
			const int new_width  = mpdp.data_transformations(i).new_width();
			const bool is_color  = mpdp.data_transformations(i).is_color();
			const bool crop_first  = mpdp.data_transformations(i).crop_first();

			string img_path = root_folder + lines_[lines_id_][i];
			DLOG(INFO) << "Loading image " << img_path << " as data #" << i << " batchID: " << item_id;
			cv::Mat cv_img;
			// If we crop first then we should load the full resolution image
			if (crop_first) {
				cv_img = ReadImageToCVMat(img_path, 0, 0, is_color);
			} else {
				cv_img = ReadImageToCVMat(img_path, new_height, new_width, is_color);
			}

			if (!cv_img.data) {
				LOG(ERROR) << "Couldn't load image " << img_path;
				continue;
			}
			cv_imgs.push_back(cv_img);
		}
		read_time += timer.MicroSeconds();

		timer.Start();
		// Apply transformations (mirror, crop...) to the image
		// For each patch for each image
		for (int iPatch = 0; iPatch < patch_count; ++iPatch) {
			float pw, ph;
			stringstream ssw;
			ssw << lines_[lines_id_][image_count + 1 + 2*iPatch];
			ssw >> pw;
			stringstream ssh;
			ssh << lines_[lines_id_][image_count + 1 + 2*iPatch + 1];
			ssh >> ph;

			for (int iImage = 0; iImage < image_count; ++iImage) {
				int i = iPatch * image_count + iImage;
				int offset = this->prefetch_data_[i]->offset(item_id);
				this->transformed_data_[i]->set_cpu_data(top_data[i] + offset);
				if (share_random_trafos && i != 0) {
					// If we share the random transformations, set the state of the current transformation to the state of the first transformation
					this->transformers_[iImage]->ResetState(*this->transformers_[0]);
				}
				const int img_height = cv_imgs[iImage].rows;
				const int img_width = cv_imgs[iImage].cols;
				int h_off = int(img_height * ph);
				int w_off = int(img_width * pw);

				// Prevent overwriting crop values in UpdateState
				this->transformers_[iImage]->SetKeepCrop(true);
				this->transformers_[iImage]->ResetCropCoords(h_off, w_off);
				DLOG(INFO) << "Transforming bottom " << iImage << "...";
				this->transformers_[iImage]->Transform(cv_imgs[iImage], &(*this->transformed_data_[i]));
			}
		}

		// Finally put the similarity to the last top...
		bool sim;
		stringstream sssim;
		sssim << lines_[lines_id_][image_count];
		sssim >> sim;

		int i = patch_count * image_count;
		int offset = this->prefetch_data_[i]->offset(item_id);
		this->transformed_data_[i]->set_cpu_data(top_data[i] + offset);
		this->transformed_data_[i]->mutable_cpu_data()[0] = int(sim);

		// Reset state if all layers share the random transformations
		if (share_random_trafos) {
			this->transformers_[0]->ResetState();
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

INSTANTIATE_CLASS(MultiImagePatchDataLayer);
REGISTER_LAYER_CLASS(MULTI_IMAGE_PATCH_DATA, MultiImagePatchDataLayer);
}  // namespace caffe

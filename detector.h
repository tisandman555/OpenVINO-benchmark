#pragma once


#include <algorithm>
#include <chrono>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <utility>

#include <ie_device.hpp>
#include <ie_plugin_config.hpp>
#include <ie_plugin_dispatcher.hpp>
#include <ie_plugin_ptr.hpp>
#include <inference_engine.hpp>
#include <inference_engine.hpp>
#include <ie_plugin_config.hpp>
#include <ie_extension.h>

#include <ext_list.hpp>


#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/opencv.hpp>

#include <cldnn/cldnn_config.hpp>

using namespace std;
using namespace InferenceEngine;
using namespace std::chrono;

typedef struct __resultbox {
	float classid;
	float confidence;
	float left;
	float right;
	float top;
	float bottom;
}resultbox;

typedef struct __Result {
	vector<resultbox> boxs;
	cv::Mat orgimg;
	cv::Size imgsize;
	int inputid;
}Result;

class Detector {
public:

	Detector();
	~Detector();
	int Init();



	bool Detect(vector<Result>& objects);
	int GetTotalTime();
	int GetBatchSize();
	//void SetBatchSize(int batch_size);
	int GetNStreams();
	int GetNIreq();

private:
	int dummy;
	string FLAGS_i; 
	string FLAGS_m;
	int FLAGS_b;
	int FLAGS_fr;
	string FLAGS_d; 
	double FLAGS_thresh;

	string device_name;
	Core ie;
	CNNNetReader netBuilder;
	CNNNetwork cnnNetwork;
	size_t batchSize;
	Precision precision;
	ExecutableNetwork exeNetwork;

	uint32_t cpu_nstreams;
	uint32_t gpu_nstreams;
	uint32_t nireq;
	//vector<InferRequest> inferRequest;
	InferRequest inferRequest[8];

	cv::VideoCapture cap;
	cv::Mat frame, frameInfer;
	int framenum;

	size_t num_channels;
	size_t image_size;
	int infer_height;
	int infer_width;

	int image_width;
	int image_height;

	InputsDataMap inputInfo;
	OutputsDataMap outputsInfo;
	std::string outputName;
	DataPtr outputInfo;
	int maxProposalCount;
	int objectSize;

	unsigned long flag_inference_done;
	//vector<std::condition_variable> condVar;
	std::condition_variable condVar[8];

	std::chrono::milliseconds time2, time1, time3,time4;
	std::chrono::milliseconds start_time[8], end_time[8];
	int total_time;

};

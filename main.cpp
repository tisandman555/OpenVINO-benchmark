#if 1

#include "detector.h"

static string  CLASSES[] = { "background",
"aeroplane", "bicycle", "bird", "boat",
"!bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "!pottedplant",
"sheep", "sofa", "!train", "tvmonitor" };

int main(int argc, char *argv[]) {
	bool ret;
	Detector detector;
	Detector* gpdetector;

	gpdetector = &detector;

	gpdetector->Init();
	cout << "Init done" << endl;

	int nireq = gpdetector->GetNIreq();
	int BatchSize = gpdetector->GetBatchSize();

	cout << "nireq = " << nireq << endl;
	cout << "BatchSize = " << BatchSize << endl;


	for (;;)
	{
		vector<Result> objects(nireq*BatchSize);
		ret = gpdetector->Detect(objects);
		if (ret == false)
		{
			break;
		}
		for (int k = 0; k < objects.size(); k++)
		{
			for (int i = 0; i < objects[k].boxs.size(); i++)
			{
				cv::rectangle(objects[k].orgimg, cv::Point(objects[k].boxs[i].left, objects[k].boxs[i].top), cv::Point(objects[k].boxs[i].right, objects[k].boxs[i].bottom), cv::Scalar(71, 99, 250), 2);
				std::stringstream ss;
				ss << CLASSES[(int)(objects[k].boxs[i].classid)] << "/" << objects[k].boxs[i].confidence;
				std::string  text = ss.str();
				cv::putText(objects[k].orgimg, text, cv::Point(objects[k].boxs[i].left, objects[k].boxs[i].top + 20), cv::FONT_HERSHEY_PLAIN, 1.0f, cv::Scalar(0, 255, 255));
			}
			//cout << "image id = " << k << " box size: " << objects[k].boxs.size() << endl;
			cv::imshow("frame", objects[k].orgimg);
			cv::waitKey(10);
		}
	}

	std::cout << "total_time = " << gpdetector->GetTotalTime() << std::endl;

	return EXIT_SUCCESS;
}
#else
/*
* Copyright (c) 2018 Intel Corporation.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
* LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
* OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
* WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

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
//#include <ie_plugin_cpp.hpp>
#include <ie_extension.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

#include <opencv2/opencv.hpp>


#include <ext_list.hpp>


using namespace std;
using namespace InferenceEngine;
using namespace std::chrono;



static string  CLASSES[] = { "background",
"aeroplane", "bicycle", "bird", "boat",
"!bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "!pottedplant",
"sheep", "sofa", "!train", "tvmonitor" };

string FLAGS_i = "0.mp4"; //"Cars1.mp4"; // 
string FLAGS_m = "c:\\temp_20151027\\cvt_model\\mobilenet-ssd-fp32\\mobilenet-ssd.xml";
int FLAGS_b = 4;
int FLAGS_fr = 256;
string FLAGS_d = "CPU";  //"MULTI:CPU,GPU"
double FLAGS_thresh = 0.4;

inline std::ostream &operator<<(std::ostream &os, const InferenceEngine::Version &version) {
	os << "\t" << version.description << " version ......... ";
	os << version.apiVersion.major << "." << version.apiVersion.minor;

	os << "\n\tBuild ........... ";
	os << version.buildNumber;

	return os;
}

inline std::ostream &operator<<(std::ostream &os, const std::map<std::string, InferenceEngine::Version> &versions) {
	for (auto && version : versions) {
		os << "\t" << version.first << std::endl;
		os << version.second << std::endl;
	}

	return os;
}

/**
* @brief Gets filename without extension
* @param filepath - full file name
* @return filename without extension
*/
static std::string fileNameNoExt(const std::string &filepath) {
	auto pos = filepath.rfind('.');
	if (pos == std::string::npos) return filepath;
	return filepath.substr(0, pos);
}

template<typename T>
static bool isImage(const T &blob) {
	auto descriptor = blob->getTensorDesc();
	if (descriptor.getLayout() != InferenceEngine::NCHW) {
		return false;
	}
	auto channels = descriptor.getDims()[1];
	return channels == 3;
}


std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> result;
	std::stringstream ss(s);
	std::string item;

	while (getline(ss, item, delim)) {
		result.push_back(item);
	}
	return result;
}
std::vector<std::string> parseDevices(const std::string& device_string) {
	std::string comma_separated_devices = device_string;
	if (comma_separated_devices.find(":") != std::string::npos) {
		comma_separated_devices = comma_separated_devices.substr(comma_separated_devices.find(":") + 1);
	}
	auto devices = split(comma_separated_devices, ',');
	for (auto& device : devices)
		device = device.substr(0, device.find("("));
	return devices;
}

int main(int argc, char *argv[]) {

	std::chrono::milliseconds time2, time1;

	std::cout << "start" << std::endl;

	int total_time = 0;

	try {

		std::cout << "1" << std::endl;

		FILE *ROIfile = fopen("ROIs.txt", "w"); // output stored here, view with ROIviewer



												// ----------------- 2. Loading the Inference Engine -----------------------------------------------------------
												// Get optimal runtime parameters for device
		std::string device_name = FLAGS_d;

		Core ie;

		if (FLAGS_d.find("CPU") != std::string::npos) {
			// Loading default CPU extensions
			ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");

		}

		/*std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;*/
		std::cout << "Device info: " << std::endl;
		std::cout << ie.GetVersions(device_name) << std::endl;

		// ----------------- 3. Reading the Intermediate Representation network ----------------------------------------

		std::cout << "Loading network files" << std::endl;

		CNNNetReader netBuilder;
		netBuilder.ReadNetwork(FLAGS_m);
		const std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
		netBuilder.ReadWeights(binFileName);

		CNNNetwork cnnNetwork = netBuilder.getNetwork();
		const InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
		if (inputInfo.empty()) {
			throw std::logic_error("no inputs info is provided");
		}

		// ----------------- 4. Resizing network to match image sizes and given batch ----------------------------------

		if (FLAGS_b != 0) {
			ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
			bool reshape = false;
			for (const InputsDataMap::value_type& item : inputInfo) {
				auto layout = item.second->getTensorDesc().getLayout();

				int batchIndex = -1;
				if ((layout == Layout::NCHW) || (layout == Layout::NCDHW) ||
					(layout == Layout::NHWC) || (layout == Layout::NDHWC) ||
					(layout == Layout::NC)) {
					batchIndex = 0;
				}
				else if (layout == CN) {
					batchIndex = 1;
				}
				if ((batchIndex != -1) && (shapes[item.first][batchIndex] != FLAGS_b)) {
					shapes[item.first][batchIndex] = FLAGS_b;
					std::cout << " default batch size = " << shapes[item.first][batchIndex] << std::endl;
					reshape = true;
				}
			}
			if (reshape) {
				std::cout << "Resizing network to batch = " << FLAGS_b << std::endl;
				cnnNetwork.reshape(shapes);
			}
		}

		const size_t batchSize = cnnNetwork.getBatchSize();
		const Precision precision = cnnNetwork.getPrecision();
		std::cout << (FLAGS_b != 0 ? "Network batch size was changed to: " : "Network batch size: ") << batchSize <<
			", precision: " << precision << std::endl;


		// ----------------- 5. Configuring input ----------------------------------------------------------------------
		for (auto& item : inputInfo) {
			std::cout << "xxx 2 " << std::endl;
			if (isImage(item.second)) {
				/** Set the precision of input data provided by the user, should be called before load of the network to the device **/
				std::cout << "xxx 3 " << std::endl;
				item.second->setPrecision(Precision::U8);
			}
		}

		// --------------------------- 6. Prepare output blobs -------------------------------------------------
		std::cout << "Preparing output blobs" << std::endl;

		OutputsDataMap outputsInfo(cnnNetwork.getOutputsInfo());

		std::string outputName;
		DataPtr outputInfo;
		for (const auto& out : outputsInfo) {
			if (out.second->getCreatorLayer().lock()->type == "DetectionOutput") {
				outputName = out.first;
				outputInfo = out.second;
			}
		}
		std::cout << "outputName: " << outputName << std::endl;

		if (outputInfo == nullptr) {
			throw std::logic_error("Can't find a DetectionOutput layer in the topology");
		}

		const SizeVector outputDims = outputInfo->getTensorDesc().getDims();

		const int maxProposalCount = outputDims[2];
		const int objectSize = outputDims[3];

		std::cout << "maxProposalCount: " << maxProposalCount << std::endl;
		std::cout << "objectSize: " << objectSize << std::endl;

		if (objectSize != 7) {
			throw std::logic_error("Output item should have 7 as a last dimension");
		}

		if (outputDims.size() != 4) {
			throw std::logic_error("Incorrect output dimensions for SSD model");
		}

		/** Set the precision of output data provided by the user, should be called before load of the network to the device **/
		outputInfo->setPrecision(Precision::FP32);

		// ----------------- 6. Setting device configuration -----------------------------------------------------------

		auto devices = parseDevices(device_name);
#if 1
		// pin threads for CPU portion of inference
		ie.SetConfig({ { CONFIG_KEY(CPU_BIND_THREAD), "YES" } }, "CPU");
		std::cout << " CPU_BIND_THREAD : YES" << std::endl;


		// for CPU execution, more throughput-oriented execution via streams
		ie.SetConfig({ { CONFIG_KEY(CPU_THROUGHPUT_STREAMS),"CPU_THROUGHPUT_AUTO" } }, "CPU");
		uint32_t cpu_nstreams = std::stoi(ie.GetConfig("CPU", CONFIG_KEY(CPU_THROUGHPUT_STREAMS)).as<std::string>());
		std::cout << "CPU_THROUGHPUT_AUTO: Number of CPU streams = " << cpu_nstreams << std::endl;
#endif


		// ----------------- 7. Loading the model to the device --------------------------------------------------------

		std::map<std::string, std::string> config = { { CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(NO) } };

		ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, device_name, config);

		// ----------------- 8. Setting optimal runtime parameters -----------------------------------------------------

		// Number of requests
		uint32_t nireq = 0;
		if (nireq == 0) {
			std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
			try {
				nireq = exeNetwork.GetMetric(key).as<unsigned int>();
			}
			catch (const details::InferenceEngineException& ex) {
				THROW_IE_EXCEPTION
					<< "Every device used with the benchmark_app should "
					<< "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
					<< "Failed to query the metric for the " << device_name << " with error:" << ex.what();
			}
		}

		std::cout << "OPTIMAL_NUMBER_OF_INFER_REQUESTS: Number of requests = " << nireq << std::endl;

		// --------------------------- 5. Create infer request -------------------------------------------------
		std::cout << "Create infer request" << std::endl;
		InferRequest inferRequest1 = exeNetwork.CreateInferRequest();
		InferRequest inferRequest2 = exeNetwork.CreateInferRequest();
		InferRequest inferRequest3 = exeNetwork.CreateInferRequest();
		InferRequest inferRequest4 = exeNetwork.CreateInferRequest();



		//open video capture
		cv::VideoCapture cap(FLAGS_i);
		if (!cap.isOpened())   // check if VideoCapture init successful
		{
			std::cout << "Could not open input file" << std::endl;
			return 1;
		}

		cv::Mat frame, frameInfer;
		int framenum = 0;

		// --------------------------- 6. Prepare input --------------------------------------------------------
		unsigned long flag_done;

		std::condition_variable condVar1;
		inferRequest1.SetCompletionCallback(
			[&] {
			std::cout << "Completed 1" << " async request execution" << std::endl;

			/* continue sample execution after last Asynchronous inference request execution */
			framenum += batchSize;
			flag_done |= 0x1;
			condVar1.notify_one();
		});

		std::condition_variable condVar2;
		inferRequest2.SetCompletionCallback(
			[&] {
			std::cout << "Completed 2" << " async request execution" << std::endl;

			/* continue sample execution after last Asynchronous inference request execution */
			framenum += batchSize;
			flag_done |= 0x2;
			condVar2.notify_one();
		});

		std::condition_variable condVar3;
		inferRequest3.SetCompletionCallback(
			[&] {
			std::cout << "Completed 3" << " async request execution" << std::endl;

			/* continue sample execution after last Asynchronous inference request execution */
			framenum += batchSize;
			flag_done |= 0x4;
			condVar3.notify_one();
		});

		std::condition_variable condVar4;
		inferRequest4.SetCompletionCallback(
			[&] {
			std::cout << "Completed 4" << " async request execution" << std::endl;

			/* continue sample execution after last Asynchronous inference request execution */
			framenum += batchSize;
			flag_done |= 0x8;
			condVar4.notify_one();
		});

		for (;;)
		{



			int infer_width, infer_height;


			for (auto & item : inputInfo) {
				Blob::Ptr inputBlob = inferRequest1.GetBlob(item.first);
				SizeVector dims = inputBlob->getTensorDesc().getDims();
				/** Fill input tensor with images. First b channel, then g and r channels **/
				size_t num_channels = dims[1];
				size_t image_size = dims[3] * dims[2];

				//std::cout << "dims[0] = " << dims[0] << std::endl;
				//std::cout << "dims[1] = " << dims[1] << std::endl;
				//std::cout << "dims[2] = " << dims[2] << std::endl;
				//std::cout << "dims[3] = " << dims[3] << std::endl;
				infer_height = dims[2];
				infer_width = dims[3];


				auto data = inputBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type *>();
				/** Iterate over all input images **/
				for (size_t image_id = 0; image_id < batchSize; ++image_id) {

					cap.read(frame);

					//cv::imshow("frame", frame);
					//cv::waitKey(1);
					if ((!frame.data) || (framenum >= FLAGS_fr)) {
						std::cout << "end of file | reach the maximum framenum" << std::endl;
						goto end_of_infernece;

					}
					cv::resize(frame, frameInfer, cv::Size(infer_width, infer_height));

					/** Iterate over all pixel in image (b,g,r) **/
					for (size_t pid = 0; pid < image_size; pid++) {
						/** Iterate over all channels **/
						for (size_t ch = 0; ch < num_channels; ++ch) {
							/**          [images stride + channels stride + pixel id ] all in bytes            **/
							data[image_id * image_size * num_channels + ch * image_size + pid] = frameInfer.at<cv::Vec3b>(pid)[ch];;
						}
					}
				}
			}

			for (auto & item : inputInfo) {
				Blob::Ptr inputBlob = inferRequest2.GetBlob(item.first);
				SizeVector dims = inputBlob->getTensorDesc().getDims();
				/** Fill input tensor with images. First b channel, then g and r channels **/
				size_t num_channels = dims[1];
				size_t image_size = dims[3] * dims[2];

				auto data = inputBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type *>();
				/** Iterate over all input images **/
				for (size_t image_id = 0; image_id < batchSize; ++image_id) {

					cap.read(frame);

					//cv::imshow("frame", frame);
					//cv::waitKey(1);
					if ((!frame.data) || (framenum >= FLAGS_fr)) {
						std::cout << "end of file | reach the maximum framenum" << std::endl;
						goto end_of_infernece;

					}
					cv::resize(frame, frameInfer, cv::Size(infer_width, infer_height));

					/** Iterate over all pixel in image (b,g,r) **/
					for (size_t pid = 0; pid < image_size; pid++) {
						/** Iterate over all channels **/
						for (size_t ch = 0; ch < num_channels; ++ch) {
							/**          [images stride + channels stride + pixel id ] all in bytes            **/
							data[image_id * image_size * num_channels + ch * image_size + pid] = frameInfer.at<cv::Vec3b>(pid)[ch];;
						}
					}
				}
			}

			for (auto & item : inputInfo) {
				Blob::Ptr inputBlob = inferRequest3.GetBlob(item.first);
				SizeVector dims = inputBlob->getTensorDesc().getDims();
				/** Fill input tensor with images. First b channel, then g and r channels **/
				size_t num_channels = dims[1];
				size_t image_size = dims[3] * dims[2];

				auto data = inputBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type *>();
				/** Iterate over all input images **/
				for (size_t image_id = 0; image_id < batchSize; ++image_id) {

					cap.read(frame);

					//cv::imshow("frame", frame);
					//cv::waitKey(1);
					if ((!frame.data) || (framenum >= FLAGS_fr)) {
						std::cout << "end of file | reach the maximum framenum" << std::endl;
						goto end_of_infernece;

					}
					cv::resize(frame, frameInfer, cv::Size(infer_width, infer_height));

					/** Iterate over all pixel in image (b,g,r) **/
					for (size_t pid = 0; pid < image_size; pid++) {
						/** Iterate over all channels **/
						for (size_t ch = 0; ch < num_channels; ++ch) {
							/**          [images stride + channels stride + pixel id ] all in bytes            **/
							data[image_id * image_size * num_channels + ch * image_size + pid] = frameInfer.at<cv::Vec3b>(pid)[ch];;
						}
					}
				}
			}

			for (auto & item : inputInfo) {
				Blob::Ptr inputBlob = inferRequest4.GetBlob(item.first);
				SizeVector dims = inputBlob->getTensorDesc().getDims();
				/** Fill input tensor with images. First b channel, then g and r channels **/
				size_t num_channels = dims[1];
				size_t image_size = dims[3] * dims[2];

				auto data = inputBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type *>();
				/** Iterate over all input images **/
				for (size_t image_id = 0; image_id < batchSize; ++image_id) {

					cap.read(frame);

					//cv::imshow("frame", frame);
					//cv::waitKey(1);
					if ((!frame.data) || (framenum >= FLAGS_fr)) {
						std::cout << "end of file | reach the maximum framenum" << std::endl;
						goto end_of_infernece;

					}
					cv::resize(frame, frameInfer, cv::Size(infer_width, infer_height));

					/** Iterate over all pixel in image (b,g,r) **/
					for (size_t pid = 0; pid < image_size; pid++) {
						/** Iterate over all channels **/
						for (size_t ch = 0; ch < num_channels; ++ch) {
							/**          [images stride + channels stride + pixel id ] all in bytes            **/
							data[image_id * image_size * num_channels + ch * image_size + pid] = frameInfer.at<cv::Vec3b>(pid)[ch];;
						}
					}
				}
			}


			flag_done = 0;



			// --------------------------- 7. Do inference ---------------------------------------------------------
			time1 = duration_cast< milliseconds >(
				system_clock::now().time_since_epoch()
				);

			//inferRequest.Infer();

			//inferRequest.StartAsync();
			//inferRequest.Wait(IInferRequest::WaitMode::RESULT_READY);

			/* Start async request for the first time */
			inferRequest1.StartAsync();
			inferRequest2.StartAsync();
			inferRequest3.StartAsync();
			inferRequest4.StartAsync();
			/* Wait all repetitions of the async request */
#if 1
			std::mutex mutex1;
			std::unique_lock<std::mutex> lock1(mutex1);
			/*condVar.wait(lock, [&] { std::cout << "Condition not ready " << std::endl; return false; }); //always false*/
			condVar1.wait(lock1, [&] {cout<< "1:"<< flag_done  <<endl; return ((flag_done & 0x1)!=0); });
			cout << "-1:" << flag_done << endl;

			std::mutex mutex2;
			std::unique_lock<std::mutex> lock2(mutex2);
			condVar2.wait(lock2, [&] {cout << "2:" << flag_done << endl; return ((flag_done & 0x2) != 0); });
			cout << "-2:" << flag_done << endl;

			std::mutex mutex3;
			std::unique_lock<std::mutex> lock3(mutex3);
			condVar3.wait(lock3, [&] {cout << "3:" << flag_done << endl; return ((flag_done & 0x4) != 0); });
			cout << "-3:" << flag_done << endl;

			std::mutex mutex4;
			std::unique_lock<std::mutex> lock4(mutex4);
			condVar4.wait(lock4, [&] {cout << "4:" << flag_done << endl; return ((flag_done & 0x8) != 0); });
			cout << "-4:" << flag_done << endl;
#else
			while (flag_done != 0xF) Sleep(1);
#endif

			time2 = duration_cast< milliseconds >(
				system_clock::now().time_since_epoch()
				);
			std::cout << "inferRequest1.Infer() Done. Time = " << (time2 - time1).count() << std::endl;
			total_time += (time2 - time1).count();



			// --------------------------- 11. Process output -------------------------------------------------------
			std::cout << "Processing output blobs" << std::endl;

			const Blob::Ptr output_blob = inferRequest1.GetBlob(outputName);
			const float* detection = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output_blob->buffer());

			std::vector<std::vector<int> > boxes(batchSize);
			std::vector<std::vector<int> > classes(batchSize);

			/* Each detection has image_id that denotes processed image */
			for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) {
				auto image_id = static_cast<int>(detection[curProposal * objectSize + 0]);
				if (image_id < 0) {
					break;
				}

				float confidence = detection[curProposal * objectSize + 2];
				auto label = static_cast<int>(detection[curProposal * objectSize + 1]);
				auto xmin = static_cast<int>(detection[curProposal * objectSize + 3] * frame.cols);
				auto ymin = static_cast<int>(detection[curProposal * objectSize + 4] * frame.rows);
				auto xmax = static_cast<int>(detection[curProposal * objectSize + 5] * frame.cols);
				auto ymax = static_cast<int>(detection[curProposal * objectSize + 6] * frame.rows);

				std::cout << "[" << curProposal << "," << label << "] element, prob = " << confidence <<
					"    (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")" << " batch id : " << image_id;

				cv::rectangle(frame, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(71, 99, 250), 2);
				std::stringstream ss;
				ss << CLASSES[(int)(label)] << "/" << confidence;
				std::string  text = ss.str();
				cv::putText(frame, text, cv::Point(xmin, ymin + 20), cv::FONT_HERSHEY_PLAIN, 1.0f, cv::Scalar(0, 255, 255));


				if (confidence > FLAGS_thresh) {
					/** Drawing only objects with >50% probability **/
					classes[image_id].push_back(label);
					boxes[image_id].push_back(xmin);
					boxes[image_id].push_back(ymin);
					boxes[image_id].push_back(xmax - xmin);
					boxes[image_id].push_back(ymax - ymin);
					std::cout << " WILL BE PRINTED!";
				}
				std::cout << std::endl;
			}


			std::cout << "framenum = " << framenum << std::endl;
#if 0
			cv::imshow("output frame", frame);
			if (cv::waitKey(1) == 27 /*ESC*/)
			{
				break;
			}
#endif
		}
	end_of_infernece:
		std::cout << "total_time = " << total_time << std::endl;
		std::cout << "My Code END HERE" << std::endl;

		fclose(ROIfile);

	}
	catch (const InferenceEngine::details::InferenceEngineException& ex) {
		std::cerr << ex.what() << std::endl;
		return EXIT_FAILURE;
	}



	return EXIT_SUCCESS;
}

#endif

#include "detector.h"

static string  CLASSES[] = { "background",
"aeroplane", "bicycle", "bird", "boat",
"!bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "!pottedplant",
"sheep", "sofa", "!train", "tvmonitor" };

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


static std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> result;
	std::stringstream ss(s);
	std::string item;

	while (getline(ss, item, delim)) {
		result.push_back(item);
	}
	return result;
}
static std::vector<std::string> parseDevices(const std::string& device_string) {
	std::string comma_separated_devices = device_string;
	if (comma_separated_devices.find(":") != std::string::npos) {
		comma_separated_devices = comma_separated_devices.substr(comma_separated_devices.find(":") + 1);
	}
	auto devices = split(comma_separated_devices, ',');
	for (auto& device : devices)
		device = device.substr(0, device.find("("));
	return devices;
}

Detector::Detector() {
	dummy = 0;

	FLAGS_i = "0.mp4"; //"Cars1.mp4"; // 
	//FLAGS_m = "c:\\temp_20151027\\cvt_model\\mobilenet-ssd-fp16\\mobilenet-ssd.xml";
	FLAGS_m = "c:\\temp_20151027\\cvt_model\\mobilenet-ssd-int8\\17_i8.xml";
	FLAGS_b = 1;
	FLAGS_fr = 256;
	FLAGS_d = "CPU";  //"MULTI:GPU,CPU"; //"GPU";  //"MULTI:CPU,GPU"; // "CPU";  //"MULTI:CPU,GPU"; //"CPU";// "MULTI:CPU,GPU"; // "GPU";  //"MULTI:CPU,GPU"
	FLAGS_thresh = 0.4;

	total_time = 0;


} 

Detector::~Detector() {
	dummy = 1;
}

int Detector::Init() {
	device_name = FLAGS_d;

	std::cout << "FLAGS_m: " << FLAGS_m << std::endl;

	if (FLAGS_d.find("CPU") != std::string::npos) {
		// Loading default CPU extensions
		ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");

	}

	/*std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;*/
	std::cout << "Device info: " << std::endl;
	std::cout << ie.GetVersions(device_name) << std::endl;

	// ----------------- 3. Reading the Intermediate Representation network ----------------------------------------

	std::cout << "Loading network files" << std::endl;


	netBuilder.ReadNetwork(FLAGS_m);
	const std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
	netBuilder.ReadWeights(binFileName);

	cnnNetwork = netBuilder.getNetwork();
	inputInfo = cnnNetwork.getInputsInfo();
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

	batchSize = cnnNetwork.getBatchSize();
	precision = cnnNetwork.getPrecision();
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

	outputsInfo = cnnNetwork.getOutputsInfo();


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

	maxProposalCount = outputDims[2];
	objectSize = outputDims[3];

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

	if (FLAGS_d == "CPU")
	{
		//for CPU inference
		// pin threads for CPU portion of inference
		ie.SetConfig({ { CONFIG_KEY(CPU_BIND_THREAD), "YES" } }, "CPU");
		//ie.SetConfig({ { CONFIG_KEY(CPU_BIND_THREAD), "NO" } }, "CPU");
		std::cout << " CPU_BIND_THREAD :" << ie.GetConfig("CPU", CONFIG_KEY(CPU_BIND_THREAD)).as<std::string>() << std::endl;


		// for CPU execution, more throughput-oriented execution via streams
		ie.SetConfig({ { CONFIG_KEY(CPU_THROUGHPUT_STREAMS),"CPU_THROUGHPUT_AUTO" } }, "CPU");
		//ie.SetConfig({ { CONFIG_KEY(CPU_THROUGHPUT_STREAMS),std::to_string(1) } }, "CPU");
		cpu_nstreams = std::stoi(ie.GetConfig("CPU", CONFIG_KEY(CPU_THROUGHPUT_STREAMS)).as<std::string>());
		std::cout << "CPU_THROUGHPUT_AUTO: Number of CPU streams = " << cpu_nstreams << std::endl;
	}
	else if (FLAGS_d == "GPU")
	{

		//for GPU inference
		ie.SetConfig({ { CONFIG_KEY(GPU_THROUGHPUT_STREAMS),"GPU_THROUGHPUT_AUTO" } }, "GPU");
		//ie.SetConfig({ { CONFIG_KEY(GPU_THROUGHPUT_STREAMS),"1" } }, "GPU");
		gpu_nstreams = std::stoi(ie.GetConfig("GPU", CONFIG_KEY(GPU_THROUGHPUT_STREAMS)).as<std::string>());
		std::cout << "GPU_THROUGHPUT_AUTO: Number of GPU streams = " << gpu_nstreams << std::endl;
	}
	else if (FLAGS_d == "MULTI:CPU,GPU")
	{
		ie.SetConfig({ { CONFIG_KEY(CPU_BIND_THREAD), "NO" } }, "CPU");
		std::cout << " CPU_BIND_THREAD :" << ie.GetConfig("CPU", CONFIG_KEY(CPU_BIND_THREAD)).as<std::string>() << std::endl;
		
		// for CPU execution, more throughput-oriented execution via streams
		ie.SetConfig({ { CONFIG_KEY(CPU_THROUGHPUT_STREAMS),"CPU_THROUGHPUT_AUTO" } }, "CPU");
		//ie.SetConfig({ { CONFIG_KEY(CPU_THROUGHPUT_STREAMS),std::to_string(1) } }, "CPU");
		cpu_nstreams = std::stoi(ie.GetConfig("CPU", CONFIG_KEY(CPU_THROUGHPUT_STREAMS)).as<std::string>());
		std::cout << "CPU_THROUGHPUT_AUTO: Number of CPU streams = " << cpu_nstreams << std::endl;

		//for GPU inference
		ie.SetConfig({ { CONFIG_KEY(GPU_THROUGHPUT_STREAMS),"GPU_THROUGHPUT_AUTO" } }, "GPU");
		//ie.SetConfig({ { CONFIG_KEY(GPU_THROUGHPUT_STREAMS),"2" } }, "GPU");
		gpu_nstreams = std::stoi(ie.GetConfig("GPU", CONFIG_KEY(GPU_THROUGHPUT_STREAMS)).as<std::string>());
		std::cout << "GPU_THROUGHPUT_AUTO: Number of GPU streams = " << gpu_nstreams << std::endl;

		ie.SetConfig({ { CLDNN_CONFIG_KEY(PLUGIN_THROTTLE), "1" } }, "GPU");
		std::cout << "CLDNN_CONFIG_KEY(PLUGIN_THROTTLE), 1" << std::endl;
	}
	else if (FLAGS_d == "MULTI:GPU,CPU")
	{
		ie.SetConfig({ { CONFIG_KEY(CPU_BIND_THREAD), "NO" } }, "CPU");
		std::cout << " CPU_BIND_THREAD :" << ie.GetConfig("CPU", CONFIG_KEY(CPU_BIND_THREAD)).as<std::string>() << std::endl;

		// for CPU execution, more throughput-oriented execution via streams
		//ie.SetConfig({ { CONFIG_KEY(CPU_THROUGHPUT_STREAMS),"CPU_THROUGHPUT_AUTO" } }, "CPU");
		ie.SetConfig({ { CONFIG_KEY(CPU_THROUGHPUT_STREAMS),std::to_string(2) } }, "CPU");
		cpu_nstreams = std::stoi(ie.GetConfig("CPU", CONFIG_KEY(CPU_THROUGHPUT_STREAMS)).as<std::string>());
		std::cout << "CPU_THROUGHPUT_AUTO: Number of CPU streams = " << cpu_nstreams << std::endl;

		//for GPU inference
		//ie.SetConfig({ { CONFIG_KEY(GPU_THROUGHPUT_STREAMS),"GPU_THROUGHPUT_AUTO" } }, "GPU");
		ie.SetConfig({ { CONFIG_KEY(GPU_THROUGHPUT_STREAMS),"2" } }, "GPU");
		gpu_nstreams = std::stoi(ie.GetConfig("GPU", CONFIG_KEY(GPU_THROUGHPUT_STREAMS)).as<std::string>());
		std::cout << "GPU_THROUGHPUT_AUTO: Number of GPU streams = " << gpu_nstreams << std::endl;

		ie.SetConfig({ { CLDNN_CONFIG_KEY(PLUGIN_THROTTLE), "1" } }, "GPU");
		std::cout << "CLDNN_CONFIG_KEY(PLUGIN_THROTTLE), 1" << std::endl;
	}


	// ----------------- 7. Loading the model to the device --------------------------------------------------------

	std::map<std::string, std::string> config = { { CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(NO) } };

	exeNetwork = ie.LoadNetwork(cnnNetwork, device_name, config);

	// ----------------- 8. Setting optimal runtime parameters -----------------------------------------------------

	// Number of requests
	nireq = 0;
	//nireq = 1;
	if (nireq == 0) {
		std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
		try {
			nireq = exeNetwork.GetMetric(key).as<unsigned int>();
			std::cout << "nireq:  = " << nireq << std::endl;
		}
		catch (const details::InferenceEngineException& ex) {
			THROW_IE_EXCEPTION
				<< "Every device used with the benchmark_app should "
				<< "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
				<< "Failed to query the metric for the " << device_name << " with error:" << ex.what();
		}
	}

	std::cout << "OPTIMAL_NUMBER_OF_INFER_REQUESTS: Number of requests = " << nireq << std::endl;

	if (nireq > 8)
	{
		exit(0);
	}
	// --------------------------- 5. Create infer request -------------------------------------------------
	std::cout << "Create infer request" << std::endl;
	//inferRequest.resize(nireq);
	//condVar.resize(nireq);
	for (int i = 0; i < nireq; i++)
	{
		inferRequest[i] = exeNetwork.CreateInferRequest();
	}

	//open video capture
	cap.open(FLAGS_i);
	if (!cap.isOpened())   // check if VideoCapture init successful
	{
		std::cout << "Could not open input file" << std::endl;
		return false;
	}

	framenum = 0;

	for (auto & item : inputInfo) {
		Blob::Ptr inputBlob = inferRequest[0].GetBlob(item.first);
		SizeVector dims = inputBlob->getTensorDesc().getDims();
		/** Fill input tensor with images. First b channel, then g and r channels **/
		num_channels = dims[1];
		image_size = dims[3] * dims[2];
		infer_height = dims[2];
		infer_width = dims[3];

		std::cout << "num_channels:" << num_channels << " image_size:" << image_size << " infer_height:" << infer_height << " infer_width:" << infer_width << std::endl;

	}

	for (int i = 0; i < nireq; i++)
	{
		if (i == 0)
		{
			inferRequest[0].SetCompletionCallback(
				[&] {
				time3 = duration_cast< milliseconds >(
					system_clock::now().time_since_epoch()
					);
				//std::cout << "Completed 0" << " async request execution" << time1.count() << " : " << time3.count() << std::endl;

				/* continue sample execution after last Asynchronous inference request execution */
				framenum += batchSize;
				flag_inference_done |= 0x1;
				//condVar[0].notify_one();
			});
		}
		else if (i == 1)
		{
			inferRequest[1].SetCompletionCallback(
				[&] {
				time3 = duration_cast< milliseconds >(
					system_clock::now().time_since_epoch()
					);

				//std::cout << "Completed 1" << " async request execution" << time1.count() << " : " << time3.count() << std::endl;

				/* continue sample execution after last Asynchronous inference request execution */
				framenum += batchSize;
				flag_inference_done |= 0x2;
				//condVar[1].notify_one();
			});
		}
		else if (i == 2)
		{
			inferRequest[2].SetCompletionCallback(
				[&] {
				time3 = duration_cast< milliseconds >(
					system_clock::now().time_since_epoch()
					);
				//std::cout << "Completed 2" << " async request execution" << time1.count() << " : " << time3.count() << std::endl;

				/* continue sample execution after last Asynchronous inference request execution */
				framenum += batchSize;
				flag_inference_done |= 0x4;
				//condVar[2].notify_one();
			});
		}
		else if (i == 3)
		{
			inferRequest[3].SetCompletionCallback(
				[&] {
				time3 = duration_cast< milliseconds >(
					system_clock::now().time_since_epoch()
					);
				//std::cout << "Completed 3" << " async request execution" << time1.count() << " : " << time3.count() << std::endl;

				/* continue sample execution after last Asynchronous inference request execution */
				framenum += batchSize;
				flag_inference_done |= 0x8;
				//condVar[3].notify_one();
			});
		}
		else if (i == 4)
		{
			inferRequest[4].SetCompletionCallback(
				[&] {
				time3 = duration_cast< milliseconds >(
					system_clock::now().time_since_epoch()
					);
				//std::cout << "Completed 4" << " async request execution" << time1.count() << " : " << time3.count() << std::endl;

				/* continue sample execution after last Asynchronous inference request execution */
				framenum += batchSize;
				flag_inference_done |= 0x10;
				//condVar[4].notify_one();
			});
		}
		else if (i == 5)
		{
			inferRequest[5].SetCompletionCallback(
				[&] {
				time3 = duration_cast< milliseconds >(
					system_clock::now().time_since_epoch()
					);
				//std::cout << "Completed 5" << " async request execution" << time1.count() << " : " << time3.count() << std::endl;

				/* continue sample execution after last Asynchronous inference request execution */
				framenum += batchSize;
				flag_inference_done |= 0x20;
				//condVar[5].notify_one();
			});
		}
		else if (i == 6)
		{
			inferRequest[6].SetCompletionCallback(
				[&] {
				time3 = duration_cast< milliseconds >(
					system_clock::now().time_since_epoch()
					);
				//std::cout << "Completed 6" << " async request execution" << time1.count() << " : " << time3.count() << std::endl;

				/* continue sample execution after last Asynchronous inference request execution */
				framenum += batchSize;
				flag_inference_done |= 0x40;
				//condVar[6].notify_one();
			});
		}
		else if (i == 7)
		{
			inferRequest[7].SetCompletionCallback(
				[&] {
				time3 = duration_cast< milliseconds >(
					system_clock::now().time_since_epoch()
					);
				//std::cout << "Completed 7" << " async request execution" << time1.count() << " : " << time3.count() << std::endl;

				/* continue sample execution after last Asynchronous inference request execution */
				framenum += batchSize;
				flag_inference_done |= 0x80;
				//condVar[7].notify_one();
			});
		}
		else
		{
			std::cout << "Out of Scope - Callback" << std::endl;
		}

#if 0
		inferRequest2.SetCompletionCallback(
			[&] {
			std::cout << "Completed 2" << " async request execution" << std::endl;

			/* continue sample execution after last Asynchronous inference request execution */
			framenum += batchSize;
			flag_inference_done |= 0x2;
			condVar2.notify_one();
		});


		inferRequest3.SetCompletionCallback(
			[&] {
			std::cout << "Completed 3" << " async request execution" << std::endl;

			/* continue sample execution after last Asynchronous inference request execution */
			framenum += batchSize;
			flag_inference_done |= 0x4;
			condVar3.notify_one();
		});


		inferRequest4.SetCompletionCallback(
			[&] {
			std::cout << "Completed 4" << " async request execution" << std::endl;

			/* continue sample execution after last Asynchronous inference request execution */
			framenum += batchSize;
			flag_inference_done |= 0x8;
			condVar4.notify_one();
		});
#endif
	}


}

bool Detector::Detect(vector<Result>& objects) {

	cout << " Detect start " << endl;
	for (int i = 0; i < nireq; i++)
	{
		//cout << " i =  " << i << endl;
		for (auto & item : inputInfo) {
			Blob::Ptr inputBlob = inferRequest[i].GetBlob(item.first);


			auto data = inputBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type *>();
			/** Iterate over all input images **/
			for (size_t image_id = 0; image_id < batchSize; ++image_id) {
				//cout << " image_id =  " << image_id << endl;

#if 0
				cap.read(frame);
				objects[i*batchSize + image_id].orgimg = frame;
#else
				cap.read(objects[i*batchSize + image_id].orgimg);
				frame = objects[i*batchSize + image_id].orgimg;
#endif
				if ((!frame.data) || (framenum >= FLAGS_fr)) {
					std::cout << "end of file | reach the maximum framenum" << std::endl;
					return false;

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
	}



	std::cout << "read file done" << std::endl;

	flag_inference_done = 0;
	// --------------------------- 7. Do inference ---------------------------------------------------------
	time1 = duration_cast< milliseconds >(
		system_clock::now().time_since_epoch()
		);

	//inferRequest.Infer();

	//inferRequest.StartAsync();
	//inferRequest.Wait(IInferRequest::WaitMode::RESULT_READY);

	/* Start async request for the first time */
	//for (int i = 0; i < nireq; i++)
	for (int i = 0; i < nireq; i++)
	{
		//int i = 3;
		//std::cout << "start async stream: " << i << std::endl;
		start_time[i] = duration_cast< milliseconds >(
			system_clock::now().time_since_epoch()
			);
		inferRequest[i].StartAsync();
	}

	time4 = duration_cast< milliseconds >(
		system_clock::now().time_since_epoch()
		);
	/* Wait all repetitions of the async request */
#if 0
	std::mutex mutex1;
	std::unique_lock<std::mutex> lock1(mutex1);
	/*condVar.wait(lock, [&] { std::cout << "Condition not ready " << std::endl; return false; }); //always false*/
	condVar1.wait(lock1, [&] {cout << "1:" << flag_inference_done << endl; return ((flag_inference_done & 0x1) != 0); });
	cout << "-1:" << flag_inference_done << endl;

	std::mutex mutex2;
	std::unique_lock<std::mutex> lock2(mutex2);
	condVar2.wait(lock2, [&] {cout << "2:" << flag_inference_done << endl; return ((flag_inference_done & 0x2) != 0); });
	cout << "-2:" << flag_inference_done << endl;

	std::mutex mutex3;
	std::unique_lock<std::mutex> lock3(mutex3);
	condVar3.wait(lock3, [&] {cout << "3:" << flag_inference_done << endl; return ((flag_inference_done & 0x4) != 0); });
	cout << "-3:" << flag_inference_done << endl;

	std::mutex mutex4;
	std::unique_lock<std::mutex> lock4(mutex4);
	condVar4.wait(lock4, [&] {cout << "4:" << flag_inference_done << endl; return ((flag_inference_done & 0x8) != 0); });
	cout << "-4:" << flag_inference_done << endl;
#else
#define PERF_TEST
#ifdef PERF_TEST
	int fps_counter = 0;
	int time_counter[8],frame_counter[8];
	for (int n = 0; n < nireq; n++)
	{
		time_counter[n] = 0;
		frame_counter[n] = 0;
	}
	while (1)
	{
		if (flag_inference_done != 0)
		{
			for (int i = 0; i < nireq; i++)
			{
				unsigned long flag = 0x1 << i;
				if (flag_inference_done&flag)
				{
					inferRequest[i].StartAsync();
					fps_counter++;
					frame_counter[i]++;
					end_time[i] = duration_cast<milliseconds>(
						system_clock::now().time_since_epoch()
						);
					//std::cout << "Infer( " << i << " ). Time = " << (end_time[i] - start_time[i]).count() << " S:"<<start_time[i].count() <<" E:"<<end_time[i].count()<<std::endl;
					time_counter[i] += (end_time[i] - start_time[i]).count();
					//std::cout << "L[ " << i << " ]: " << (end_time[i] - start_time[i]).count() << std::endl;
					flag_inference_done = flag_inference_done & (~flag);
					start_time[i] = end_time[i];
					//std::cout << "S( " << i << " ):" << start_time[i].count() << std::endl;
					if ((fps_counter % 100) == 0)
					{
						time4 = duration_cast<milliseconds>(
							system_clock::now().time_since_epoch()
							);
						std::cout << "FPS: " << fps_counter * FLAGS_b * 1000 / (time4 - time1).count() << std::endl;

						for (int n = 0; n < nireq; n++)
						{
							if (frame_counter[n] != 0)
							{
								std::cout << "Latency[ " << n << " ]: " << time_counter[n] * 1000 * FLAGS_b / frame_counter[n] << "  " << "frame_counter= " << frame_counter[n] << std::endl;
							}
							time_counter[n] = 0;
							frame_counter[n] = 0;
						}
						time1 = time4;
						fps_counter = 0;
					}
				}
			}
		}

	}
#else
	/*while (flag_inference_done != 0xFF) Sleep(1);*/
	while (flag_inference_done != 0xFF) Sleep(1);
#endif
#endif

	time2 = duration_cast< milliseconds >(
		system_clock::now().time_since_epoch()
		);
	std::cout << "inferRequest.Infer() all Done. Time = " << (time4 - time1).count() << " : " << (time2 - time1).count() << std::endl;
	total_time += (time2 - time1).count();

	// --------------------------- 11. Process output -------------------------------------------------------
	std::cout << "Processing output blobs" << std::endl;

	for (int i = 0; i < nireq; i++)
	{
		const Blob::Ptr output_blob = inferRequest[i].GetBlob(outputName);
		const float* detection = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output_blob->buffer());

		//std::vector<std::vector<int> > boxes(batchSize);
		//std::vector<std::vector<int> > classes(batchSize);

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

			//cv::rectangle(frame, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(71, 99, 250), 2);
			std::stringstream ss;
			ss << CLASSES[(int)(label)] << "/" << confidence;
			std::string  text = ss.str();
			//cv::putText(frame, text, cv::Point(xmin, ymin + 20), cv::FONT_HERSHEY_PLAIN, 1.0f, cv::Scalar(0, 255, 255));


			if (confidence > FLAGS_thresh) {
				/** Drawing only objects with >50% probability **/
				//classes[image_id].push_back(label);
				//boxes[image_id].push_back(xmin);
				//boxes[image_id].push_back(ymin);
				//boxes[image_id].push_back(xmax - xmin);
				//boxes[image_id].push_back(ymax - ymin);
				std::cout << " WILL BE PRINTED!";

				resultbox object;
				object.classid = (int)label;
				object.confidence = confidence;
				object.left = (int)xmin;
				object.top = (int)ymin;
				object.right = (int)xmax;
				object.bottom = (int)ymax;
				if ((label != 5) && (label != 16))
				{
					objects[i*batchSize + image_id].boxs.push_back(object);
					std::cout << " classid: " << label << " WILL BE ADDED!";
				}

			}
			std::cout << std::endl;
		}
	}


	std::cout << "framenum = " << framenum << std::endl;



	return true;
}

int Detector::GetTotalTime()
{
	return total_time;
}

int Detector::GetBatchSize()
{
	return batchSize;
}

//void Detector::SetBatchSize(int batch_size)
//{
//	FLAGS_b = batchSize;
//}

int Detector::GetNStreams()
{
	return cpu_nstreams;
}
int Detector::GetNIreq()
{
	return nireq;
}
# OpenVINO-benchmark
based on OpenVINO 2019R3 benchmark_app

基于openvino 2019R3的benchmark_app改写简写，去掉了命令传递参数的方法，所有参数改为代码里hard code;去掉了智能指针之类的高级用法，只使用简单的操作系统提供的多线程同步接口
编译时需要link FFMPEG，如果需要看显示结果需要link OpenCV

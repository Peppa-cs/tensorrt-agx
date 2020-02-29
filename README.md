
Tensorrt provides some samples to help beginners.The test.cpp references sampleMNIST/sampleINT8/sampleINT8API/sampleGoogleNet/trtexec and code on Github.

The test/test.cpp(my code) is used to test imagenet data and trtexec/trtexec.cpp(tensorrt sample) is used to test random data.

#Command Line performs Tensorrt using Caffe FrameWork #

#./test --batch  --weightsFileName --prototxtFileName --int8/--fp16 --Network --engine --calibfile#

--batch means the number of pictures in a batch
--weightsFileName XX.caffemodel --prototxtFileName XX.protorxr
--int8/--fp16 the data type of tensor
--Network the name of NN
--engine As building engine is time consuming, it can be serialized a plan and deserialized to an engine.
--calibfile The process of calibration is also time consuming,it can be saved for later running.

Firstly, create builder to create "network" and "config".
The "config" is used to set some flags,for example,config->setAvgTimingIterations(8);
The "network" is NN construction after parser parses weight file and deployment. 

If the int8 mode is choosed, tensorrt need run almost 500 pictures to calibrate  precision.
//------about INT8 from Tensorrt Documents------------//
In order to perform INT8 inference, FP32 activation tensors and weights need to be quantized. In order to represent 32-bit floating point values and INT 8-bit quantized values, TensorRT needs to understand the dynamic range of each activation tensor. The dynamic range is used to determine the appropriate quantization scale.

TensorRT supports symmetric quantization with a quantization scale calculated using absolute maximum dynamic range values.

TensorRT needs the dynamic range for each tensor in the network. There are two ways in which the dynamic range can be provided to the network:
1): manually set the dynamic range for each network tensor using setDynamicRange API.

2): use INT8 calibration to generate per tensor dynamic range using the calibration dataset.

The dynamic range API can also be used along with INT8 calibration, such that manually setting the range will take precedence over the calibration generated a dynamic range. Such a scenario is possible if INT8 calibration does not generate a satisfactory dynamic range for certain tensors。
//-----------------------------------------------------//
I have used two ways to get dynamic range. First using calibration dataset to get calibration table, it also has been saved for futher use. And i use way 1 by reading the data in calibration table to manually set the dynamic range.But it seems that  way 2(reading saved calibration table) is faster than way 1(manually set the dynamic range).

If the fp16 mode is choosed, the parameter of caffe parser need to be set to kHALF besides setting the builder flag. #parser->parse(... DataType::kHALF; The kFLOAT flag are required to calibrate precision when setting int8 mode.The kHALF is float16 type and the kFLOAT is float32 type.

Secondly, the inference will be performed by using "enqueue" function. The input data will be transferred from host to gpu and the outpub probability will be back to verify the correction.

The CalibrationTableAlexnet is a calibration table of Alexnet.



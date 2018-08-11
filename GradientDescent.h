#pragma once

#include <iostream>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

class GradientDescent
{
public:
	GradientDescent();
	void vanilla_update(Mat& W, Mat& b, Mat dW, Mat db, float learning_rate = 0.01);
	void vanilla_update(UMat& W, UMat& b, UMat dW, UMat db, float learning_rate = 0.01);
};


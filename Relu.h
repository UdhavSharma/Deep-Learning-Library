#pragma once

#include <iostream>
#include <opencv2\opencv.hpp>
#include "Layers.h"

using namespace std;
using namespace cv;

class Relu : public Layers
{
public:
	Relu(float slopes);
	Mat Forward(Mat input);
	Mat Backward(Mat dout);
	UMat Forward(UMat input);
	UMat Backward(UMat dout);
	bool hasweights() { return false; }
private:
	float slope;
	Mat X;
	UMat UX;
};


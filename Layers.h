#pragma once

#include <iostream>
#include "opencv2\opencv.hpp"

using namespace std;
using namespace cv;

class Layers
{
public:
	Layers();
	virtual Mat getweight() { return Mat(); }
	virtual Mat Forward(Mat input) { return Mat(); }
	virtual Mat Backward(Mat dout) { return Mat();  }
	virtual UMat Forward(UMat input) { return UMat(); }
	virtual UMat Backward(UMat dout) { return UMat(); }
	virtual void saveweights(String sw, String sb) {}
	virtual void loadweights(String sw, String sb) {}
	virtual bool hasweights() { return false; }
	virtual void vanilla_update(float learning_rate, float lam) {}
};


#pragma once
#include <iostream>
#include <opencv2\opencv.hpp>
#include "Layers.h"

using namespace std;
using namespace cv;

class FullyConnectedLayer : public Layers
{
public:
	FullyConnectedLayer(int input, int output);
	UMat Forward(UMat input);
	UMat Backward(UMat dout);
	bool hasweights() { return true; }
	void saveweights(String sw, String sb);
	void loadweights(String sw, String sb);
	Mat getweight() { Mat Wm; W.copyTo(Wm); return Wm; }
	void vanilla_update(float learning_rate, float lam);
	void setWeight(UMat Wg) { W.release(); W = Wg; }
	void setBias(UMat bg) { b.release();  b = bg; }
private:
	UMat X;
	UMat W;
	UMat b;
	UMat db;
	UMat dW;
};


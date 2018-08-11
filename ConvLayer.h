#pragma once
#include <iostream>
#include <opencv2\opencv.hpp>
#include "Layers.h"

using namespace std;
using namespace cv;

class ConvLayer : public Layers {
public:
	ConvLayer(int rows, int cols, int depth, int no_of_filterss, int stride, int pad);
	void saveweights(String sw, String sb);
	void loadweights(String sw, String sb);
	bool hasweights() { return true; }
	Mat Forward(Mat input);
	Mat Backward(Mat dout);
	Mat getweight() { Mat Wx; W.copyTo(Wx); return Wx; }
	void vanilla_update(float learning_rate, float lam);
	void setWeight(Mat Wg) { W.release(); W = Wg; }
	void setBias(Mat bg) { b.release();  b = bg; }
private:
	int r, c, d, no_of_filters, stride, pad;
	int ri, ci, batch;
	Mat W;
	Mat b;
	UMat X_cols;
	UMat W_cols;
	Mat dW;
	Mat db;
	void initialize_weights();
};

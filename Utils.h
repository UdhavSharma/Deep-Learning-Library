#pragma once

#include <iostream>
#include <opencv2\opencv.hpp>
#include "Layers.h"
#include "Flatten.h"
#include "Softmax.h"
#include "BinaryCvMat.h"

using namespace std;
using namespace cv;

class Utils
{
public:
	Utils();
	void shuffleRows(Mat &X, Mat &Y);
	void makedataset(String path, int image_size, String save, bool tests = false, bool verbose = true);
	void loaddataset(Mat& X, Mat& Y, String path, String save);
	UMat Forward(vector<Layers*> Convlayers, Flatten& f, vector<Layers*> FClayers, Mat X_mini);
	void Backward(vector<Layers*> Convlayers, Flatten& f, vector<Layers*> FClayers, UMat dout);
	void update(vector<Layers*> Convlayers, vector<Layers*> FClayers, float learning_rate, float lam = 0.001);
	void saveweights(vector<Layers*> Convlayers, vector<Layers*> FClayers, vector<string> swc, vector<string> sbc, vector<string> swf, vector<string> sbf);
	void loadweights(vector<Layers*> Convlayers, vector<Layers*> FClayers, vector<string> swc, vector<string> sbc, vector<string> swf, vector<string> sbf);
	void train(Mat X, Mat Y, int batch_size, int epochs, int learning_rate, int lam, vector<Layers*> Convlayers, Flatten f, vector<Layers*> FClayers, bool weightupdate = true, bool verbose = true);
	void test(Mat X, Mat Y, int batch_size, vector<Layers*> Convlayers, Flatten f, vector<Layers*> FClayers);
	float l2_regularization(vector<Layers*> Convlayers, vector<Layers*> FClayers, float lam = 0.001);
	void delta_l2_regularization(Mat& dW, Mat W, float lam);
	void delta_l2_regularization(UMat& dW, UMat W, float lam);
	Mat im2col(Mat input, int r, int c, int stride_r, int stride_c);
	Mat im2col3D(Mat input, int r, int c, int stride_r, int stride_c);
	Mat col2im(Mat input, int r, int c, int stride_r, int stride_c, int ri, int ci, int d);
	Mat col2im3D(Mat input, int r, int c, int stride_r, int stride_c, int batch, int ri, int ci, int d);
};

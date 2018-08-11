#pragma once

#include <iostream>
#include <opencv2\opencv.hpp>
#include "Layers.h"

using namespace std;
using namespace cv;

class Utils
{
public:
	Utils();
	float l2_regularization(vector<Layers*> Convlayers, vector<Layers*> FClayers, float lam = 0.001);
	void delta_l2_regularization(Mat& dW, Mat W, float lam);
	void delta_l2_regularization(UMat& dW, UMat W, float lam);
	Mat im2col(Mat input, int r, int c, int stride_r, int stride_c);
	Mat im2col3D(Mat input, int r, int c, int stride_r, int stride_c);
	Mat col2im(Mat input, int r, int c, int stride_r, int stride_c, int ri, int ci, int d);
	Mat col2im3D(Mat input, int r, int c, int stride_r, int stride_c, int batch, int ri, int ci, int d);
};


#pragma once
#include <iostream>
#include <opencv2\opencv.hpp>
#include "Layers.h"

using namespace std;
using namespace cv;

class Pooling : public Layers
{
public:
	Pooling(int f, int s, int p);
	Mat Forward(Mat input);
	Mat Backward(Mat dout);
	bool hasweights() { return false; }
private:
	int xcolr, xcolc;
	vector<Mat> index;
	int batch, ri, ci;
	int filter;
	int stride;
	int pad;
};


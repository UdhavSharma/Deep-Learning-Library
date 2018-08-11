#pragma once

#include <iostream>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

class Flatten
{
public:
	Flatten();
	UMat Forward(Mat input);
	Mat Backward(UMat dout);
	bool hasweights() { return false; }
private:
	int n, r, c, d;
};


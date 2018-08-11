#pragma once
#include <iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Softmax {
public:
	Softmax();
	float softmax_loss(Mat scores, Mat y);
	float accuracy();
	Mat softmax_back();
private:
	Mat prob;
	Mat correct_class;
};


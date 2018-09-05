#pragma once

//Credits and source: https://github.com/takmin/BinaryCvMat
//Modified to fit library needs

#include <iostream>
#include <fstream>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

class BinaryCvMat
{
public:
	BinaryCvMat();
	bool SaveMatBinary(const std::string& filename, const cv::Mat& output);
	bool LoadMatBinary(const std::string& filename, cv::Mat& output);
private:
	bool writeMatBinary(std::ofstream& ofs, const cv::Mat& out_mat);
	bool readMatBinary(std::ifstream& ifs, cv::Mat& in_mat);
};


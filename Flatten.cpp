#include "Flatten.h"

using namespace std;
using namespace cv;

Flatten::Flatten()
{
}

UMat Flatten::Forward(Mat input) {
	n = input.size[0];
	r = input.size[1];
	c = input.size[2];
	d = input.channels();

	//int sx[] = { n,r,c };
	//Mat flats = Mat::zeros(3, sx, CV_32FC(d));

	//float* p1 = flats.ptr<float>(0);
	//float* p2 = input.ptr<float>(0);

	//int count = 0;

	//for (int i = 0; i < n; i++) {
	//	for (int ch = 0; ch < d; ch++) {
	//		for (int j = 0; j < r; j++) {
	//			for (int k = 0; k < c; k++) {
	//				p1[count++] = p2[i * r * c * d + j * c * d + k * d + ch];
	//			}
	//		}
	//	}
	//}

	UMat flat;
	input.copyTo(flat);

	int s[] = { input.size[0], input.size[1] * input.size[2] * input.channels() };
	flat = flat.reshape(1, 2, s).t();

	return flat;
}

Mat Flatten::Backward(UMat dout) {
	assert(n == dout.cols && r*c*d == dout.rows);
	Mat dX;
	dout.copyTo(dX);
	
	int s[] = { n, r, c };
	dX = dX.t();
	dX = dX.reshape(d, 3, s);

	//Mat dXs = Mat::zeros(3, s, CV_32FC(d));

	//float* p1 = dXs.ptr<float>(0);
	//float* p2 = dX.ptr<float>(0);

	//int count = 0;

	//for (int i = 0; i < n; i++) {
	//	for (int ch = 0; ch < d; ch++) {
	//		for (int j = 0; j < r; j++) {
	//			for (int k = 0; k < c; k++) {
	//				p1[i * r * c * d + j * c * d + k * d + ch] = p2[count++];
	//			}
	//		}
	//	}
	//}

	return dX;
}

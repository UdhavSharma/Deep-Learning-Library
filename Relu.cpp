#include "Relu.h"

using namespace std;
using namespace cv;

Relu::Relu(float slopes)
{
	slope = slopes;
}

Mat Relu::Forward(Mat input) {
	X = input;

	if (input.channels() > 1) {
		int s[] = { input.size[0], input.size[1] * input.size[2] * input.channels() };
		X = X.reshape(1, 2, s);
	}

	Mat out = Mat::ones(X.rows, X.cols, X.type());
	out.setTo(slope, X <= 0);
	multiply(out, X, out);

	if (input.channels() > 1) {
		int s[] = { input.size[0], input.size[1], input.size[2] };
		out = out.reshape(input.channels(), 3, s);
	}

	return out;
}

Mat Relu::Backward(Mat dout) {
	Mat dX = dout;

	if (dout.channels() > 1) {
		int s[] = { dout.size[0], dout.size[1] * dout.size[2] * dout.channels() };
		dX = dX.reshape(1, 2, s);
	}

	assert(X.rows == dX.rows && X.cols == dX.cols);

	Mat out = Mat::ones(dX.rows, dX.cols, dX.type());
	out.setTo(slope, X <= 0);
	multiply(out, dX, out);
	
	if (dout.channels() > 1) {
		int s[] = { dout.size[0], dout.size[1], dout.size[2] };
		out = out.reshape(dout.channels(), 3, s);
	}
	X.release();
	return out;
}

UMat Relu::Forward(UMat input) {
	input.copyTo(UX);

	if (input.channels() > 1) {
		int s[] = { input.size[0], input.size[1] * input.size[2] * input.channels() };
		UX = UX.reshape(1, 2, s);
	}

	UMat out = UMat::ones(UX.rows, UX.cols, UX.type());
	UMat mask;
	compare(UX, 0, mask, CMP_LE);
	out.setTo(slope, mask);
	multiply(out, UX, out);

	if (input.channels() > 1) {
		int s[] = { input.size[0], input.size[1], input.size[2] };
		out = out.reshape(input.channels(), 3, s);
	}

	return out;
}

UMat Relu::Backward(UMat dout) {
	UMat dX; dout.copyTo(dX);

	if (dout.channels() > 1) {
		int s[] = { dout.size[0], dout.size[1] * dout.size[2] * dout.channels() };
		dX = dX.reshape(1, 2, s);
	}

	assert(UX.rows == dX.rows && UX.cols == dX.cols);

	UMat out = UMat::ones(dX.rows, dX.cols, dX.type());
	UMat mask;
	compare(UX, 0, mask, CMP_LE);
	out.setTo(slope, mask);
	multiply(out, dX, out);

	if (dout.channels() > 1) {
		int s[] = { dout.size[0], dout.size[1], dout.size[2] };
		out = out.reshape(dout.channels(), 3, s);
	}
	UX.release();
	return out;
}


#include "Pooling.h"
#include "Utils.h"

using namespace std;
using namespace cv;

Pooling::Pooling(int f, int s, int p){
	filter = f;
	stride = s;
	pad = p;
}

Mat Pooling::Forward(Mat input) {

	Mat inputs;
	Mat output;
	Utils a;

	//Padding 3D

	int sp[] = { input.size[0],input.size[1] + (2 * pad), input.size[2] + (2 * pad) };
	inputs = Mat::zeros(3, sp, input.type());
	Range ranges[3];
	ranges[0] = { 0,input.size[0] };
	ranges[1] = { pad,input.size[1] + pad };
	ranges[2] = { pad,input.size[2] + pad };
	input.copyTo(inputs(ranges));

	assert((inputs.size[1] - filter) % stride == 0);
	assert((inputs.size[2] - filter) % stride == 0);

	batch = inputs.size[0];
	ri = inputs.size[1];
	ci = inputs.size[2];

	//3D output

	vector<Mat> channels;
	vector<Mat> ochannels;
	split(inputs, channels);

	int ro = ((inputs.size[1] - filter) / stride) + 1;
	int co = ((inputs.size[2] - filter) / stride) + 1;
	int s[] = { inputs.size[0], ro, co };

	for (int i = 0; i < inputs.channels(); i++) {
		Mat X_col = a.im2col3D(channels[i], filter, filter, stride, stride);
		ochannels.push_back(Mat(X_col.rows,1,CV_32FC1));
		index.push_back(Mat(X_col.rows,1,CV_32FC1));
		float* p1 = X_col.ptr<float>(0);
		float* p2 = ochannels[i].ptr<float>(0);
		float* p3 = index[i].ptr<float>(0);
		for (int j = 0; j < X_col.rows; j++) {
			float max = p1[j * X_col.cols]; int maxindex = 0;
			for (int k = 0; k < X_col.cols; k++) {
				if (p1[j * X_col.cols + k] > max) {
					max = p1[j * X_col.cols + k]; maxindex = k;
				}
			}
			p2[j] = max;
			p3[j] = maxindex;
		}
		ochannels[i] = ochannels[i].reshape(0, 3, s);
		xcolr = X_col.rows;
		xcolc = X_col.cols;
	}

	merge(ochannels, output);

	return output;
}

Mat Pooling::Backward(Mat dout) {
	assert(dout.size[0] * dout.size[1] * dout.size[2] == xcolr);

	Utils a;
	vector<Mat> doutchannels;
	vector<Mat> dX_col;
	split(dout, doutchannels);

	for (int i = 0; i < dout.channels(); i++) {
		int s[] = { xcolr, 1 };
		doutchannels[i] = doutchannels[i].reshape(1, 2, s);
		dX_col.push_back(Mat::zeros(xcolr, xcolc, CV_32FC1));
		float* p1 = dX_col[i].ptr<float>(0);
		float* p2 = index[i].ptr<float>(0);
		float* p3 = doutchannels[i].ptr<float>(0);
		for (int j = 0; j < dX_col[i].rows; j++) {
			int k = p2[j];
			p1[j * dX_col[i].cols + k] = p3[j];
		}
		dX_col[i] = a.col2im3D(dX_col[i], filter, filter, stride, stride, batch, ri, ci, 1);
	}

	Mat dXpad;
	merge(dX_col, dXpad);

	//Relase Memory
	index.clear();

	//Removing Padding

	if (pad == 0) return dXpad;

	int s[] = { dXpad.size[0], dXpad.size[1] - (2 * pad), dXpad.size[2] - (2 * pad) };
	Mat dX = Mat::zeros(3, s, dXpad.type());
	Range ranges[3];
	ranges[0] = { 0,dX.size[0] };
	ranges[1] = { pad,dX.size[1] + pad };
	ranges[2] = { pad,dX.size[2] + pad };
	dXpad(ranges).copyTo(dX);

	return dX;
}

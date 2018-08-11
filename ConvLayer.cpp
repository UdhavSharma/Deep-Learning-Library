#include "ConvLayer.h"
#include "Utils.h"
#include "BinaryCvMat.h"
#include "GradientDescent.h"

using namespace std;
using namespace cv;

ConvLayer::ConvLayer(int rows, int cols, int depth, int no_of_filterss, int strides, int pads) {
	r = rows;
	c = cols;
	d = depth;
	no_of_filters = no_of_filterss;
	stride = strides;
	pad = pads;
	initialize_weights();
}

void ConvLayer::initialize_weights() {
	const int s[] = {no_of_filters, r, c};
	const int sz[] = {no_of_filters, r*c*d};

	W.create(3, s, CV_32FC(d));
	W = W.reshape(1, 2, sz);
	//randu(W, Scalar::all(-0.01), Scalar::all(0.01));
	//randn(W, 0.0, sqrt(2.0 / (r*c*d))); //He Initialization
	//randn(W, 0.0, 1 / sqrt(no_of_filters)); //Xaviers Initialization
	randn(W, 0.0, 0.05);
	W = W.reshape(d, 3, s);
	b = Mat::zeros(1, no_of_filters, CV_32FC1);
}

void ConvLayer::saveweights(String sw, String sb) {
	BinaryCvMat bm;
	int s[] = { no_of_filters,r,c };
	int ss[] = { no_of_filters,r*c };
	W = W.reshape(0, 2, ss);
	bm.SaveMatBinary(sw, W);
	bm.SaveMatBinary(sb, b);
	W = W.reshape(0, 3, s);
}

void ConvLayer::loadweights(String sw, String sb) {
	BinaryCvMat bm;
	int s[] = { no_of_filters,r,c };
	int ss[] = { no_of_filters,r*c };
	W = W.reshape(0, 2, ss);
	bm.LoadMatBinary(sw, W);
	bm.LoadMatBinary(sb, b);
	W = W.reshape(0, 3, s);
}

void ConvLayer::vanilla_update(float learning_rate, float lam) {
	Utils a;
	GradientDescent g;
	a.delta_l2_regularization(dW, W, lam);
	int s[] = { W.size[0], W.size[1] * W.size[2] * W.channels() };
	g.vanilla_update(W, b, dW, db, learning_rate);
	dW.release();
	db.release();
}

Mat ConvLayer::Forward(Mat input) {

	assert(input.channels() == d);

	Mat output;
	Mat inputs;
	Utils a;

	//Padding 3D

	int s[] = { input.size[0],input.size[1] + (2 * pad), input.size[2] + (2 * pad) };
	inputs = Mat::zeros(3,s,input.type());
	Range ranges[3];
	ranges[0] = { 0,input.size[0] };
	ranges[1] = { pad,input.size[1] + pad };
	ranges[2] = { pad,input.size[2] + pad };
	input.copyTo(inputs(ranges));

	assert((inputs.size[1] - r) % stride == 0);
	assert((inputs.size[2] - c) % stride == 0);

	if(inputs.type() == CV_16SC(d)) inputs.convertTo(inputs, CV_32FC(d));
	if (inputs.type() == CV_8UC(d)) inputs.convertTo(inputs, CV_32FC(d));

	batch = inputs.size[0];
	ri = inputs.size[1];
	ci = inputs.size[2];

	//3D output

	Mat X_col = a.im2col3D(inputs, r, c, stride, stride);
	int sw[] = { no_of_filters, r*c*d };
	Mat W_col = W.reshape(1, 2, sw);

	int ro = ((inputs.size[1] - r) / stride) + 1;
	int co = ((inputs.size[2] - c) / stride) + 1;

	output.create(input.size[0]*ro*co, no_of_filters, CV_32FC1);

	UMat outputs; X_col.copyTo(X_cols); W_col.copyTo(W_cols); output.copyTo(outputs);

	gemm(X_cols, W_cols, 1.0, noArray(), 0.0, outputs, GEMM_2_T);

	outputs.copyTo(output);

	for (int i = 0; i < output.rows; i++) {
		output.row(i) += b;
	}

	int so[] = { input.size[0], ro, co };
	output = output.reshape(no_of_filters, 3, so);

	return output;
}

Mat ConvLayer::Backward(Mat dout) {
	assert(dout.channels() == no_of_filters);
	Utils a;

	int sww[] = { no_of_filters, r*c*d };

	int so[] = { dout.size[0] * dout.size[1] * dout.size[2], dout.channels() };
	dout = dout.reshape(1, 2, so);

	//db
	
	reduce(dout, db, 0, CV_REDUCE_SUM);

	//dW
	
	UMat douts; UMat dWs; dout.copyTo(douts);
	
	//dW_col
	gemm(douts, X_cols, 1.0, noArray(), 0.0, dWs, GEMM_1_T);
	dWs.copyTo(dW);

	//Reshape dW_cols to dW
	int sw[] = {no_of_filters, r, c};
	dW = dW.reshape(d, 3, sw);

	//dX

	UMat dX_cols; Mat dX_col;
	
	//dX_col
	gemm(douts, W_cols, 1.0, noArray(), 0.0, dX_cols);
	dX_cols.copyTo(dX_col);

	//Release memory
	X_cols.release(); W_cols.release();

	//Convert dX_col to dX after padding
	Mat dXpad = a.col2im3D(dX_col, r, c, stride, stride, batch, ri, ci, d);

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

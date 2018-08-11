#include "FullyConnectedLayer.h"
#include "Utils.h"
#include "BinaryCvMat.h"
#include "GradientDescent.h"

using namespace std;
using namespace cv;

FullyConnectedLayer::FullyConnectedLayer(int input, int output)
{
	//X = UMat::zeros(input, batchsize, CV_32FC1);
	W = UMat::zeros(input, output, CV_32FC1);
	//randu(W, Scalar::all(-0.01), Scalar::all(0.01));
	randn(W, 0.0, sqrt(2.0 / input)); //He Initialization
	//randn(W, 0.0, 1 / sqrt(input));
	//randn(W, 0.0, 0.01);
	b = UMat::zeros(output, 1, CV_32FC1);
}

void FullyConnectedLayer::saveweights(String sw, String sb) {
	BinaryCvMat bm;
	Mat Wmat; Mat bmat; W.copyTo(Wmat); b.copyTo(bmat);
	bm.SaveMatBinary(sw, Wmat);
	bm.SaveMatBinary(sb, bmat);
}

void FullyConnectedLayer::loadweights(String sw, String sb) {
	BinaryCvMat bm;
	Mat Wmat; Mat bmat;
	bm.LoadMatBinary(sw, Wmat);
	bm.LoadMatBinary(sb, bmat);
	Wmat.copyTo(W); bmat.copyTo(b);
}

void FullyConnectedLayer::vanilla_update(float learning_rate, float lam) {
	Utils a; 
	GradientDescent g;
	a.delta_l2_regularization(dW, W, lam);
	g.vanilla_update(W, b, dW, db, learning_rate);
	dW.release();
	db.release();
}

UMat FullyConnectedLayer::Forward(UMat input) {
	
	assert(input.rows == W.rows);
	
	UMat output;
	Utils a;

	X = input;
	if (X.type() == CV_16SC1) X.convertTo(X, CV_32FC1);
	//Forward Propogation

	//W.t * x + b
	gemm(W, X, 1.0, noArray(), 0.0, output, GEMM_1_T);

	//make bias from output neurons x 1 to output neurons x batch size, repeat the column batchsize times to get same dimensions

	UMat bias; UMat bx = UMat::ones(1, X.cols, CV_32FC1);
	gemm(b, bx, 1.0, noArray(), 0.0, bias);

	add(output, bias, output);

	return output;
}

UMat FullyConnectedLayer::Backward(UMat dout) {
	assert(dout.rows == W.cols && dout.cols == X.cols);

	//db

	//Sum of same node gradient accross each input
	reduce(dout, db, 1, CV_REDUCE_SUM);

	//dW

	//X * dout.t
	gemm(X, dout, 1.0, noArray(), 0.0, dW, GEMM_2_T);

	//dX
	UMat dX;

	//W * dout
	gemm(W, dout, 1.0, noArray(), 0.0, dX);

	return dX;
}

#include "GradientDescent.h"


GradientDescent::GradientDescent()
{
}

void GradientDescent::vanilla_update(Mat& W, Mat& b, Mat dW, Mat db, float learning_rate) {
	W += -learning_rate * dW;
	b += -learning_rate * db;
}

void GradientDescent::vanilla_update(UMat& W, UMat& b, UMat dW, UMat db, float learning_rate) {
	multiply(dW, learning_rate, dW);
	subtract(W, dW, W);
	multiply(db, learning_rate, db);
	subtract(b, db, b);
}
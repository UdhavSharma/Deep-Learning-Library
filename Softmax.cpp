#include "softmax.h"

Softmax::Softmax() {}

float Softmax::softmax_loss(Mat scores, Mat y) {
	/*
	X_dims : NxD
	W_dims : DxC
	scores dim: NxC
	*/
	//initialize loss to 0
	scores = scores.t();
	float loss = 0.0;
	//removing numerical instability
	for (int i = 0; i < scores.rows; i++) {
		//calc max from each row and subtract the max value elementwise from each row
		double min, max;
		minMaxLoc(scores.row(i), &min, &max);
		scores.row(i) = scores.row(i) - (float)max;
	}
	//raising each element to the power of e
	exp(scores, scores);
	//computing denominator; s_i dims: Nx1
	Mat s_i;
	reduce(scores, s_i, 1, CV_REDUCE_SUM, -1);
	//computing numerator; p_i dims: NxC
	Mat p_i;
	Mat s_temp;
	repeat(s_i, 1, scores.cols, s_temp);
	divide(scores, s_temp, p_i);
	prob = p_i;
	correct_class = y;
	//log likelihood of correct class
	for (int i = 0; i < p_i.rows; i++) {
		int row_idx = i;
		int col_idx = y.at<int>(i);
		float elem = p_i.at<float>(row_idx, col_idx);
		loss += -1 * log(elem);
	}
	loss = loss / scores.rows;
	//cout << loss << endl;
	return loss;
}

float Softmax::accuracy() {
	float* p = prob.ptr<float>(0);
	float count = 0;
	for (int i = 0; i < prob.rows; i++) {
		float max = 0; int maxindex = 0;
		for (int j = 0; j < prob.cols; j++) {
			if (p[i * prob.cols + j] > max) {
				max = p[i * prob.cols + j];
				maxindex = j;
			}
		}

		if (maxindex == correct_class.at<int>(i)) {
			count++;
		}
	}

	float accuracy = count / (float)prob.rows;
	return accuracy;
}

Mat Softmax::softmax_back() {
	Mat grad;
	for (int i = 0; i < prob.rows; i++) {
		int row_idx = i;
		int col_idx = correct_class.at<int>(i);
		prob.at<float>(row_idx, col_idx) -= 1.0f;
	}

	grad = prob/(prob.rows);
	return grad.t();
}
#include "Utils.h"

using namespace std;
using namespace cv;

Utils::Utils()
{
}

float Utils::l2_regularization(vector<Layers*> Convlayers, vector<Layers*> FClayers, float lam) {
	float reg_loss = 0.0;
	for (int i = 0; i < Convlayers.size(); i++) {
		if (Convlayers[i]->hasweights()) {
			Mat W = Convlayers[i]->getweight();
			int s[] = { W.size[0], W.size[1] * W.size[2] * W.channels() };
			W = W.reshape(1, 2, s);
			multiply(W, W, W);
			float sums = (float)(sum(W)[0]);
			reg_loss += 0.5f * lam * sums;
		}
	}

	for (int i = 0; i < FClayers.size(); i++) {
		if (FClayers[i]->hasweights()) {
			Mat W = FClayers[i]->getweight();
			multiply(W, W, W);
			float sums = (float)(sum(W)[0]);
			reg_loss += 0.5f * lam * sums;
		}
	}

	return reg_loss;
}

void Utils::delta_l2_regularization(Mat& dW, Mat W, float lam) {
	dW += lam * W;
}

void Utils::delta_l2_regularization(UMat& dW, UMat W, float lam) {
	UMat Wx;
	multiply(W, lam, Wx);
	add(dW, Wx, dW);
}

Mat Utils::im2col(Mat input, int r, int c, int stride_r, int stride_c) {

	int m = input.rows;
	int n = input.cols;
	int d = input.channels();

	int ro = (m - r) / stride_r + 1;
	int co = (n - c) / stride_c + 1;

	cv::Mat out = cv::Mat::zeros(ro*co, r*c*d, CV_32FC1);

	float* pi;
	float* po;

	for (int i = 0, input_col = 0; i< co; i++, input_col += stride_c) {
		for (int j = 0, input_row = 0; j< ro; j++, input_row += stride_r) {
			int rowIdx = i + j * co;
			po = out.ptr<float>(rowIdx);
			for (int yy = 0; yy < c; yy++) {
				for (int xx = 0; xx < r; xx++) {
					int colIdx = (xx * c + yy) * d;
					pi = input.ptr<float>(input_row + xx);
					for (int channel = 0; channel < d; channel++) {
						//out.at<float>(rowIdx, colIdx + channel) = input.at<Vec3f>(input_row + xx, input_col + yy)[channel];
						po[colIdx + channel] = pi[(input_col + yy) * d + channel];
					}
				}
			}
		}
	}

	return out;
}


Mat Utils::im2col3D(Mat input, int r, int c, int stride_r, int stride_c) {

	int m = input.size[1];
	int n = input.size[2];
	int d = input.channels();
	int batch = input.size[0];

	int ro = (m - r) / stride_r + 1;
	int co = (n - c) / stride_c + 1;

	cv::Mat out = cv::Mat::zeros(ro*co*batch, r*c*d, CV_32FC1);

	float* pi;
	float* po;

	for (int k = 0; k < batch; k++) {
		pi = input.ptr<float>(k);
		for (int i = 0, input_col = 0; i< co; i++, input_col += stride_c) {
			for (int j = 0, input_row = 0; j< ro; j++, input_row += stride_r) {
				int rowIdx = (i + j * co) + (ro * co * k);
				po = out.ptr<float>(rowIdx);
				for (int yy = 0; yy < c; yy++) {
					for (int xx = 0; xx < r; xx++) {
						int colIdx = (xx * c + yy) * d;
						//pi = input.ptr<float>(input_row + xx);
						for (int channel = 0; channel < d; channel++) {
							//out.at<float>(rowIdx, colIdx + channel) = input.at<Vec3f>(input_row + xx, input_col + yy)[channel];
							po[colIdx + channel] = pi[(input_row + xx) * n * d + (input_col + yy) * d + channel];
						}
					}
				}
			}
		}
	}

	return out;
}

Mat Utils::col2im(Mat input, int r, int c, int stride_r, int stride_c, int ri, int ci, int d) {

	int ro = (ri - r) / stride_r + 1;
	int co = (ci - c) / stride_c + 1;

	cv::Mat out = cv::Mat::zeros(ri, ci, CV_32FC(d));

	float* pi;
	float* po;

	for (int i = 0, input_col = 0; i< co; i++, input_col += stride_c) {
		for (int j = 0, input_row = 0; j< ro; j++, input_row += stride_r) {
			int rowIdx = i + j * co;
			po = input.ptr<float>(rowIdx);
			for (int yy = 0; yy < c; yy++) {
				for (int xx = 0; xx < r; xx++) {
					int colIdx = (xx * c + yy) * d;
					pi = out.ptr<float>(input_row + xx);
					for (int channel = 0; channel < d; channel++) {
						//out.at<float>(rowIdx, colIdx + channel) = input.at<Vec3f>(input_row + xx, input_col + yy)[channel];
						pi[(input_col + yy) * d + channel] += po[colIdx + channel];
					}
				}
			}
		}
	}

	return out;
}


Mat Utils::col2im3D(Mat input, int r, int c, int stride_r, int stride_c, int batch, int ri, int ci, int d) {

	int ro = (ri - r) / stride_r + 1;
	int co = (ci - c) / stride_c + 1;

	int s[] = { batch,ri,ci };
	cv::Mat out = cv::Mat::zeros(3, s, CV_32FC(d));

	float* pi;
	float* po;

	for (int k = 0; k < batch; k++) {
		pi = out.ptr<float>(k);
		for (int i = 0, input_col = 0; i< co; i++, input_col += stride_c) {
			for (int j = 0, input_row = 0; j< ro; j++, input_row += stride_r) {
				int rowIdx = (i + j * co) + (ro * co * k);
				po = input.ptr<float>(rowIdx);
				for (int yy = 0; yy < c; yy++) {
					for (int xx = 0; xx < r; xx++) {
						int colIdx = (xx * c + yy) * d;
						//pi = input.ptr<float>(input_row + xx);
						for (int channel = 0; channel < d; channel++) {
							//out.at<float>(rowIdx, colIdx + channel) = input.at<Vec3f>(input_row + xx, input_col + yy)[channel];
							pi[(input_row + xx) * ci * d + (input_col + yy) * d + channel] += po[colIdx + channel];
						}
					}
				}
			}
		}
	}

	return out;
}
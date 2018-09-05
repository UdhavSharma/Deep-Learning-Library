#include "Utils.h"

using namespace std;
using namespace cv;

Utils::Utils()
{
}

void Utils::shuffleRows(Mat &X, Mat &Y)
{
	vector <int> seeds;
	for (int cont = 0; cont < X.rows; cont++)
		seeds.push_back(cont);

	randShuffle(seeds);

	Mat outputX; Mat outputY;
	for (int cont = 0; cont < X.rows; cont++) {
		outputX.push_back(X.row(seeds[cont]));
		outputY.push_back(Y.row(seeds[cont]));
	}

	outputX.copyTo(X);
	outputY.copyTo(Y);
}

void Utils::makedataset(String path, int image_size, String save, bool tests, bool verbose) {
	BinaryCvMat b;

	//String path = "F:\\Dataset\\train";
	//String save = "train";

	Mat X; Mat Y;

	vector<String> gn;
	glob(path, gn, true);
	for (int i = 0; i < gn.size(); i++) {
		vector<String> fn;
		glob(gn[i], fn, true);
		for (int j = 0; j < fn.size(); j++) {
			Mat im = imread(fn[j]);
			Size size(image_size, image_size);
			resize(im, im, size);
			int s[] = { 1, image_size * image_size * 3 };
			im = im.reshape(1, 2, s);
			X.push_back(im);
			Y.push_back(i);
			if (verbose) {
				cout << X.rows << endl;
				cout << i << endl << endl;
			}
		}
	}

	shuffleRows(X, Y);

	//Mat X_train; Mat Y_train;
	//Mat X_test; Mat Y_test;

	//for (int i = 0; i < 3800; i++) {
	//	X_train.push_back(X.row(i));
	//	Y_train.push_back(Y.row(i));
	//}

	//for (int i = 3800; i < X.rows; i++) {
	//	X_test.push_back(X.row(i));
	//	Y_test.push_back(Y.row(i));
	//}

	Mat mean;
	reduce(X, mean, 0, CV_REDUCE_AVG);

	X.convertTo(X, CV_16SC1);
	//X_test.convertTo(X_test, CV_16SC1);

	if (!tests) {
		for (int i = 0; i < X.rows; i++) {
			X.row(i) -= mean;
		}
	}

	//for (int i = 0; i < X_test.rows; i++) {
	//	X_test.row(i) -= mean;
	//}

	String x = path + "\\X_" + save + ".bin";
	String y = path + "\\Y_" + save + ".bin";

	cout << "Writing X" << endl;
	b.SaveMatBinary(x, X);
	cout << "Writing labels(Y)" << endl;
	b.SaveMatBinary(y, Y);
	//cout << "Writing X_test" << endl;
	//b.SaveMatBinary("F:\\CS\\C++ Projects\\Image\\Image\\Dataset\\X_test_segmented.bin", X_test);
	//cout << "Writing Y_test" << endl;
	//b.SaveMatBinary("F:\\CS\\C++ Projects\\Image\\Image\\Dataset\\Y_test_segmented.bin", Y_test);
}

void Utils::loaddataset(Mat& X, Mat& Y, String path, String save) {
	BinaryCvMat b;
	//String path = "F:\\Dataset\\train";
	//String save = "train"; //exclude X_ and Y_ and .bin

	String x = path + "\\X_" + save + ".bin";
	String y = path + "\\Y_" + save + ".bin";
	b.LoadMatBinary(x, X);
	b.LoadMatBinary(y, Y);
	//b.LoadMatBinary("F:\\CS\\C++ Projects\\Image\\Image\\Dataset\\X_test_segmented.bin", X_test);
	//b.LoadMatBinary("F:\\CS\\C++ Projects\\Image\\Image\\Dataset\\Y_test_segmented.bin", Y_test);
}

UMat Utils::Forward(vector<Layers*> Convlayers, Flatten& f, vector<Layers*> FClayers, Mat X_mini) {
	Mat out = X_mini;
	for (int i = 0; i < Convlayers.size(); i++) {
		out = Convlayers[i]->Forward(out);
	}

	UMat out1 = f.Forward(out);

	for (int i = 0; i < FClayers.size(); i++) {
		out1 = FClayers[i]->Forward(out1);
	}
	return out1;
}

void Utils::Backward(vector<Layers*> Convlayers, Flatten& f, vector<Layers*> FClayers, UMat dout) {
	UMat back = dout;
	for (int i = (int)FClayers.size() - 1; i >= 0; i--) {
		back = FClayers[i]->Backward(back);
	}

	Mat back1 = f.Backward(back);

	for (int i = (int)Convlayers.size() - 1; i >= 0; i--) {
		back1 = Convlayers[i]->Backward(back1);
	}
}

void Utils::update(vector<Layers*> Convlayers, vector<Layers*> FClayers, float learning_rate, float lam) {
	for (int i = 0; i < Convlayers.size(); i++) {
		if (Convlayers[i]->hasweights()) {
			Convlayers[i]->vanilla_update(learning_rate, lam);
		}
	}

	for (int i = 0; i < FClayers.size(); i++) {
		if (FClayers[i]->hasweights()) {
			FClayers[i]->vanilla_update(learning_rate, lam);
		}
	}
}

void Utils::saveweights(vector<Layers*> Convlayers, vector<Layers*> FClayers, vector<string> swc, vector<string> sbc, vector<string> swf, vector<string> sbf) {
	for (int i = 0; i < Convlayers.size(); i++) {
		if (Convlayers[i]->hasweights()) {
			Convlayers[i]->saveweights(swc[i], sbc[i]);
		}
	}

	for (int i = 0; i < FClayers.size(); i++) {
		if (FClayers[i]->hasweights()) {
			FClayers[i]->saveweights(swf[i], sbf[i]);
		}
	}
}

void Utils::loadweights(vector<Layers*> Convlayers, vector<Layers*> FClayers, vector<string> swc, vector<string> sbc, vector<string> swf, vector<string> sbf) {
	for (int i = 0; i < Convlayers.size(); i++) {
		if (Convlayers[i]->hasweights()) {
			Convlayers[i]->loadweights(swc[i], sbc[i]);
		}
	}

	for (int i = 0; i < FClayers.size(); i++) {
		if (FClayers[i]->hasweights()) {
			FClayers[i]->loadweights(swf[i], sbf[i]);
		}
	}
}

void Utils::train(Mat X, Mat Y, int batch_size, int epochs, int learning_rate, int lam, vector<Layers*> Convlayers, Flatten f, vector<Layers*> FClayers, bool weightupdate, bool verbose) {
	for (int i = 0; i < epochs; i++) {
		cout << "Epoch " << i + 1 << ": ";
		shuffleRows(X, Y);
		for (int j = 0, k = 0; j < X.rows / batch_size; j++, k += batch_size) {
			Mat X_mini; Mat Y_mini;
			for (int s = k; s < k + batch_size; s++) {
				if (s >= X.rows) break;
				X_mini.push_back(X.row(s));
				Y_mini.push_back(Y.row(s));
			}
			X_mini.convertTo(X_mini, CV_32FC1);
			X_mini /= 255;
			Softmax soft;
			int rc = (int)sqrt(X_mini.cols / 3);
			int s[] = { X_mini.rows, rc, rc };

			X_mini = X_mini.reshape(3, 3, s);

			UMat out = Forward(Convlayers, f, FClayers, X_mini);

			Mat scores; out.copyTo(scores);

			float loss = soft.softmax_loss(scores, Y_mini);
			float accuracy = soft.accuracy();
			Mat out1 = soft.softmax_back();

			//cout << "Normal loss: " << loss << endl;

			loss += l2_regularization(Convlayers, FClayers);
			if (verbose) {
				cout << "Iteration " << j + 1 << ": Loss: " << loss << " Accuracy: " << accuracy << endl;
			}

			if (weightupdate) {
				UMat dout; out1.copyTo(dout);
				Backward(Convlayers, f, FClayers, dout);
				update(Convlayers, FClayers, learning_rate, lam);
			}
			X_mini.release(); Y_mini.release();
		}
		if (verbose) cout << endl;
	}
}

void Utils::test(Mat X, Mat Y, int batch_size, vector<Layers*> Convlayers, Flatten f, vector<Layers*> FClayers) {
	cout << "Testing: " << endl << endl;

	vector<float> accuracies;
	for (int j = 0, k = 0; j < X.rows / batch_size; j++, k += batch_size) {
		Mat X_mini; Mat Y_mini;
		for (int s = k; s < k + batch_size; s++) {
			if (s >= X.rows) break;
			X_mini.push_back(X.row(s));
			Y_mini.push_back(Y.row(s));
		}
		X_mini.convertTo(X_mini, CV_32FC1);
		X_mini /= 255;
		Softmax soft;
		int rc = (int)sqrt(X_mini.cols / 3);
		int s[] = { X_mini.rows, rc, rc };

		X_mini = X_mini.reshape(3, 3, s);

		UMat out = Forward(Convlayers, f, FClayers, X_mini);

		Mat scores; out.copyTo(scores);

		float loss = soft.softmax_loss(scores, Y_mini);
		float accuracy = soft.accuracy();

		accuracies.push_back(accuracy);

		//cout << "Normal loss: " << loss << endl;

		loss += l2_regularization(Convlayers, FClayers);
		cout << "Iteration " << j + 1 << ": Loss: " << loss << " Accuracy: " << accuracy << endl;

		X_mini.release(); Y_mini.release();
	}

	float avgacc = 0;
	for (int i = 0; i < accuracies.size(); i++) {
		avgacc += accuracies[i];
	}

	avgacc /= accuracies.size();

	cout << "Test Accuracy: " << avgacc << endl;
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

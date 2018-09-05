#include<iostream>
#include<opencv2\opencv.hpp> //OPENCV
#include "Layers.h" // Base Class
//include all the layers used
#include "ConvLayer.h"
#include "Flatten.h"
#include "FullyConnectedLayer.h"
#include "Pooling.h"
#include "Relu.h"
#include "Utils.h" //import utilities

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	Utils a; //To access utilities

	//First make bin files of dataset for faster access. This process will take sometime to finish.
	//Bin files are stored in the same folder as dataset

	//makedataset(path of data, image_size, name of saved file, true(if test data) default false, true(verbose))

	//Path of data is the folder containing all the different categories of images stored in separate folders.

	//Examples of making train and test datasets.
	a.makedataset("F:\\Dataset\\train",28,"train");
	a.makedataset("F:\\Dataset\\test",28,"test",true);

	//Making dataset is only required once unless there are changes made in images.

	//Declare variables to store images and labels.

	Mat X_train; //Images
	Mat Y_train; //Labels
	Mat X_test;
	Mat Y_test;

	//Load the bin files
	//loaddataset(Images, Labels, path of data, name of saved file)
	//Do not include X_ or Y_ and .bin in the name of saved file

	a.loaddataset(X_train, Y_train, "F:\\Dataset\\train", "train");
	a.loaddataset(X_test, Y_test, "F:\\Dataset\\test", "test");

	//Normalizing the test data with the mean of train data
	Mat mean;
	reduce(X_train, mean, 0, CV_REDUCE_AVG);

	for (int i = 0; i < X_test.rows; i++) {
		X_test.row(i) -= mean;
	}

	int batch_size = 100; //Mini Batch Size
	float slope = 0.0f; //Slope of relu non linearity for leaky relu
	float learning_rate = 0.01f; //Learning Rate
	float lam = 0.01f; //Regularizatio Rate
	int epochs = 600; //Number of epochs
	
	//Architecture

	//Simple architecture example

	ConvLayer cvl1(5, 5, 3, 6, 1, 2); //Convolutional Layer (filter_h, filter_w, filter_depth, no_of_filters, stride, padding)
	Relu r1(slope); //Relu Layer (slope)
	Pooling p1(2, 2, 0); //Max Pooling Layer (filter_size, stride, padding)
	ConvLayer cvl2(5, 5, 6, 16, 1, 0);
	Relu r2(slope);
	Pooling p2(2, 2, 0);
	Flatten f; //Flatten
	FullyConnectedLayer fc1(400, 120); //Fully Connected Layer (input_nodes, output_nodes)
	Relu r3(slope);
	FullyConnectedLayer fc2(120, 84);
	Relu r4(slope);
	FullyConnectedLayer fc3(84, 12);

	//Make separate vectors for convlayers and fclayers and also make string vectors to save there weights and biases

	vector<Layers*> Convlayers;		vector<string> swc;		vector<string> sbc;
	Convlayers.push_back(&cvl1);	swc.push_back("F:\\Models\\1.0\\cvl1w.bin"); sbc.push_back("F:\\Models\\1.0\\cvl1b.bin");
	Convlayers.push_back(&r1);		swc.push_back("NILL"); sbc.push_back("NILL"); //Push back NILL for layers with no weights and biases
	Convlayers.push_back(&p1);		swc.push_back("NILL"); sbc.push_back("NILL");
	Convlayers.push_back(&cvl2);	swc.push_back("F:\\Models\\1.0\\cvl2w.bin"); sbc.push_back("F:\\Models\\1.0\\cvl2b.bin");
	Convlayers.push_back(&r2);		swc.push_back("NILL"); sbc.push_back("NILL");
	Convlayers.push_back(&p2);		swc.push_back("NILL"); sbc.push_back("NILL");

	//Push convlayers in convlayers vector and push fclayers and after in fclayers vector, no need to push flatten

	vector<Layers*> FClayers;		vector<string> swf;		vector<string> sbf;
	FClayers.push_back(&fc1);		swf.push_back("F:\\Models\\1.0\\fc1w.bin");	sbf.push_back("F:\\Models\\1.0\\fc1b.bin");
	FClayers.push_back(&r3);		swf.push_back("NILL"); sbf.push_back("NILL");
	FClayers.push_back(&fc2);		swf.push_back("F:\\Models\\1.0\\fc2w.bin");	sbf.push_back("F:\\Models\\1.0\\fc2b.bin");
	FClayers.push_back(&r4);		swf.push_back("NILL"); sbf.push_back("NILL");
	FClayers.push_back(&fc3);		swf.push_back("F:\\Models\\1.0\\fc3w.bin");	sbf.push_back("F:\\Models\\1.0\\fc3b.bin");
	
	//Load weights if using previous model or testing
	//cout << "Loading Weights: " << endl;
	//a.loadweights(Convlayers, FClayers, swc, sbc, swf, sbf);
	//cout << "Weights laoded" << endl;

	//Training
	a.train(X_train, Y_train, batch_size, epochs, learning_rate, lam, Convlayers, f, FClayers, false);

	//Saving weights
	cout << "Saving Weights: " << endl;
	a.saveweights(Convlayers, FClayers, swc, sbc, swf, sbf);
	cout << "Weights saved" << endl;

	a.test(X_test, Y_test, batch_size, Convlayers, f, FClayers);

	getchar();
	return 0;
}

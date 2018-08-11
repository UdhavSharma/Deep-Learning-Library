# Deep-Learning-Library
A deep learning library made in C++ using OpenCV used to create CNN architectures.\n
Requires OpenCV to be pre-installed.\n
OpenCV's UMat data type is used for matrix multiplication which makes use of GPU for faster calculations.

#  Layers

1. Convolutional Layer (filter_h, filter_w, filter_depth, no_of_filters, stride, padding)
2. Max Pooling Layer (filter_size, stride, padding)
3. Relu Layer (slope)
4. Flatten
5. Fully Connected Layer (input_nodes, output_nodes)
6. Softmax (Scores, Labels)

# How To Use

1. Each layer has its separate file. Include to use that layer.
2. Layer details are provided while creating that layers object.
3. Check out test.cpp file to learn how to fully use this library from using dataset and getting the final model.

# Authors and Maintainers

1. Udhav Sharma (https://www.github.com/UdhavSharma)
2. Shubham Bhatnagar (https://github.com/shubham-bhatnagar)

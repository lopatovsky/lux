#ifndef EIGEN_CPP
#define EIGEN_CPP

#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/src/IterativeSolvers/util/ConvolutionUtils>

#include <vector>
#include <cmath>

#include "utils.cpp"

using namespace std;
using namespace Eigen;

MatrixXf gaussian_kernel(int kernel_size, float sigma){
    MatrixXf kernel = MatrixXf::Zero(kernel_size, kernel_size);
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            float x = i - kernel_size/2;
            float y = j - kernel_size/2;
            kernel(i, j) = exp(-(x*x + y*y)/(2*sigma*sigma));
        }
    }
    kernel /= kernel.sum();
    return kernel;
}

std::vector<std::vector<int>> convolution( std::vector<std::vector<float>>& input,
                                           const MatrixXf& kernel){
    // Convert the input vector to an Eigen matrix
    MatrixXf mat = Map<MatrixXf>(input[0].data(), input.size(), input[0].size());

    int padding_size = kernel.rows()/2;

    // Add padding to the input matrix
    int padded_rows = mat.rows() + 2 * padding_size;
    int padded_cols = mat.cols() + 2 * padding_size;

    MatrixXf padded_mat = MatrixXf::Zero(padded_rows, padded_cols);
    padded_mat.block(padding_size, padding_size, mat.rows(), mat.cols()) = mat;

    // Perform convolution with the kernel
    ConvolutionOptions<float> conv_options;
    conv_options.setPadding(padding_size, padding_size);
    conv_options.setOutputSize(mat.rows(), mat.cols());
    MatrixXf conv = Eigen::Convolution<MatrixXf, MatrixXf, ConvolutionOptions<float>>(padded_mat, kernel, conv_options);

    // Convert the convolution result back to a vector
    std::vector<std::vector<int>> output(conv.rows(), std::vector<int>(conv.cols()));
    for (int i = 0; i < conv.rows(); i++) {
        for (int j = 0; j < conv.cols(); j++) {
            output[i][j] = static_cast<int>(conv(i, j));
        }
    }

    return output;
}

std::vector<std::vector<int>> convolution(const std::vector<std::vector<int>>& input,
                                           const MatrixXf& kernel){
    std::vector<std::vector<float>> output(input.size(), std::vector<float>(input[0].size()));
    for (int i = 0; i < input.size(); i++) {
        for (int j = 0; j < input[0].size(); j++) {
            output[i][j] = static_cast<float>(input[i][j]);
        }
    }
    convolution(output, kernel);
}


int test( vvi input ){


    cerr << "eigen" << endl;
    return 3;
}

//// eigen_tensor_convolve_multiple_filters.cpp
//#include <chrono>
//#include <iostream>
//#include <eigen3/unsupported/Eigen/CXX11/Tensor>
//
//
//int main() {
//    Eigen::Tensor<float, 3> input(128, 1024, 1024);
//    Eigen::Tensor<float, 3> filter(128, 3, 3);
//
//    using namespace std::chrono;
//    for (std::size_t run = 0; run < 5; ++run) {
//        const auto start_time_ns = high_resolution_clock::now().time_since_epoch().count();
//
//        Eigen::array<ptrdiff_t, 3> dims({0, 1, 2});
//        Eigen::Tensor<float, 3> output = input.convolve(filter, dims);
//
//        const auto end_time_ns = high_resolution_clock::now().time_since_epoch().count();
//        const auto elapsed_s = ((end_time_ns - start_time_ns) / 1000000) / 1000.0;
//        std::cout << "Convolution took " << elapsed_s << " s." << std::endl;
//    }
//}

#endif
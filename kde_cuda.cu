// kde_cuda.cu - Implementacao com CUDA em C++

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

const int WIDTH = 512, HEIGHT = 512;
const float BANDWIDTH = 1.0f;

__device__ float gaussian_kernel(float x, float y, float bandwidth) {
    return expf(-(x*x + y*y) / (2 * bandwidth * bandwidth)) / (2 * M_PI * bandwidth * bandwidth);
}

__global__ void kde_kernel(unsigned char *input, float *output, int width, int height, float bandwidth) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < height && j < width) {
        float density = 0.0f;
        for (int k = 0; k < height; k++) {
            for (int l = 0; l < width; l++) {
                float x_diff = j - l;
                float y_diff = i - k;
                density += input[k * width + l] * gaussian_kernel(x_diff, y_diff, bandwidth);
            }
        }
        output[i * width + j] = density;
    }
}

int main() {
    Mat img = imread("../image.png", IMREAD_GRAYSCALE);
    resize(img, img, Size(WIDTH, HEIGHT));
    
    unsigned char *d_input;
    float *d_output;
    
    cudaMalloc(&d_input, WIDTH * HEIGHT);
    cudaMalloc(&d_output, WIDTH * HEIGHT * sizeof(float));
    cudaMemcpy(d_input, img.data, WIDTH * HEIGHT, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);
    kde_kernel<<<gridSize, blockSize>>>(d_input, d_output, WIDTH, HEIGHT, BANDWIDTH);
    
    float output[WIDTH * HEIGHT];
    cudaMemcpy(output, d_output, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
    
    FileStorage file("output_cuda.yml", FileStorage::WRITE);
    file << "output" << Mat(HEIGHT, WIDTH, CV_32F, output);
    file.release();
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    cout << "Resultados salvos em 'output_cuda.yml'." << endl;
    return 0;
}

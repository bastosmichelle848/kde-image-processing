# KDE Image Processing

This project performs image processing using Kernel Density Estimation (KDE) and other advanced techniques. The goal is to apply transformations and image analysis with the help of libraries like OpenCV, CUDA, and OpenMP.

## Features

- Image processing with OpenCV.
- Kernel Density Estimation (KDE) using CUDA and OpenMP.
- Reading, resizing, and manipulating images.
- Generating results in formats like `.yml`.

## Prerequisites

Before you begin, make sure you have the following software installed:

- **CUDA** (compatible version for your system).
- **OpenMP** (if using OpenMP implementation).
- **OpenCV** (version 4.5 or higher).
- **CMake** for project configuration.
- **Python3** for running the Python-based KDE.

You can install OpenCV, CUDA, and OpenMP from the following links:

- [OpenCV](https://opencv.org/)
- [CUDA](https://developer.nvidia.com/cuda-toolkit)
- [OpenMP](https://www.openmp.org/)

## How to Run

### 1. Kernel Density Estimation (Python Version)

To run the KDE using Python:

python3 kernel_density_estimation4.py

### 2. Kernel Density Estimation (OpenMP Implementation)

2.1. Compile the code using make (after configuring with CMake):
make
2.2. Run the OpenMP version:

./kernel_density_estimation_open_mp4

### 3. Kernel Density Estimation (CUDA Implementation)

3.1. Compile the CUDA code:

nvcc -o kde_cuda kde_cuda.cu

3.2. Run the CUDA version:

./kde_cuda

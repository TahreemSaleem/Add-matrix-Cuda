#include "cuda.h"
#include "stdio.h"
__global__
void addSquareMatrix (int *A, int *B, int *result, int n) {
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(x < n && y < n) {
		result[y * n + x] = A[y * n + x] + B[y * n + x];
		//The same as: result[y][x] = arr1[y][x] + arr2[y][x];
	}
}

int main() {
	cudaEvent_t start, stop;

	float t;

	cudaEventCreate(&start);

	cudaEventRecord(start, 0);
	const int N = 15000;

	int *mat1_h = (int *)malloc(sizeof(int) * N * N);
	int *mat2_h = (int *)malloc(sizeof(int) * N * N);

	int *mat1_d, *mat2_d, *result_d;
	cudaMalloc(&mat1_d, sizeof(int) * N * N);
	cudaMalloc(&mat2_d, sizeof(int) * N * N);
	cudaMalloc(&result_d, sizeof(int) * N * N);

	//cudaMemcpy(mat1_d, mat1_h, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	//cudaMemcpy(mat2_d, mat2_h, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	dim3 dimBlock(256, 256);
	dim3 dimGrid(N/256, N/256);

	addSquareMatrix<<<dimGrid, dimBlock>>>(mat1_d, mat2_d, result_d, N);

	int *result_h = (int *)malloc(sizeof(int) * N);
	//cudaMemcpy(result_h, result_d, sizeof(int) * N, cudaMemcpyDeviceToHost);

	//print results

	cudaFree(result_d);
	cudaFree(mat1_d);
	cudaFree(mat2_d);
	free(mat1_h);
	free(mat2_h);
	free(result_h);
	cudaEventCreate(&stop);

	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop); 

	cudaEventElapsedTime(&t, start, stop);
	printf("Time for the kernel: %f ms\n", t);
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "hip/hip_runtime.h"
typedef struct {
	int width;
	int height;
	int stride;
	int* elements;
} Matrix;
//线程块大小
#define BLOCK_SIZE 16
//矩阵大小
#define MATRIX_SIZE 5760
void constantInit(int *elements, int size)
{
	for (int i = 0; i < size; ++i)
	{
		elements[i] = (int)(rand() % 10 );
	}
}
void printMatrix(Matrix *matrix)
{
	int index;
	int matrixSize = matrix->height*matrix->width;
	for (int i = 0; i < matrix->height; i++)
	{
		for (int j = 0; j < matrix->width; j++)
		{
			index = i * (matrix->width) + j;
			//printf("%2d ", matrix->elements[index]);
		}
		//printf("\n");
	}
	//printf("\n");
}
__device__ void SetElement(const Matrix A, int row, int col, double value)
{
	A.elements[row * A.stride + col] = value;
}
__device__ double GetElement(const Matrix A, int row, int col)
{
	return A.elements[row * A.stride + col];
}


//获取A的BLOCK_SIZE*BLOCK_SIZE的子矩阵ASUB
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
	Matrix Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.stride = A.stride;

	Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
	printf("Asub.elements:%d\n",A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col]);
	return Asub;
}

__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C)
{
	//将行和列分块
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	
	//每个线程块计算C的一个子矩阵Csub
	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
	
	//每个线程计算Csub中的一个元素，通过将结果累加到Cvalue
	double Cvalue = 0;
	
	//Csub中的线程行列     
	int row = threadIdx.y;
	int col = threadIdx.x;
	
	//循环遍历A和B的所有子矩阵，以计算Csub,并对每个子矩阵相乘，将结果累加
	for (int m = 0; m < (A.width / BLOCK_SIZE); ++m)
	{		
		printf("blockRow %d\n",blockRow);
		printf("blockCol %d\n", blockCol);	

		//获取A B 的子矩阵Asub Bsub
		Matrix Asub = GetSubMatrix(A, blockRow, m);
		Matrix Bsub = GetSubMatrix(B, m, blockCol);
		
		//分别用于保存子矩阵Asub Bsub的共享内存
		__shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];
		
		//将Asub Bsub加载到共享内存中，每个线程加载每个子矩阵的一个元素
		As[row][col] = GetElement(Asub, row, col);
		Bs[row][col] = GetElement(Bsub, row, col);
		__syncthreads();
		for (int e = 0; e < BLOCK_SIZE; ++e)
		{
			Cvalue += As[row][e] * Bs[e][col];	
		}
		__syncthreads();
	}
	
	//将Csub写入设备内存 ，每个线程写一个元素
	SetElement(Csub, row, col, Cvalue);
}
int main(int argc, char **argv)
{
	clock_t start = 0, finish = 0;
	int devID = 0;
	hipSetDevice(devID);
	hipDeviceProp_t deviceProp;
	hipGetDevice(&devID);
	hipGetDeviceProperties(&deviceProp, devID);
	printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
		devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	
	Matrix A, B, C;
	A.width = A.height = A.stride = MATRIX_SIZE;
	B.width = B.height = B.stride = MATRIX_SIZE;
	C.width = C.height = C.stride = MATRIX_SIZE;
	
	int size_A = A.width * A.height;
	int size_B = B.width * B.height;
	int size_C = C.width * C.height;
	
	A.elements = (int *)malloc(sizeof(int) * size_A);
	B.elements = (int *)malloc(sizeof(int) * size_B);
	C.elements = (int *)malloc(sizeof(int) * size_C);
	
	constantInit(A.elements, size_A);
	constantInit(B.elements, size_B);
	printf("   Matrix_A: (%d×%d)\n", A.height, A.width);
	printMatrix(&A);
	printf("   Matrix_B: (%d×%d)\n", B.height, B.width);
	printMatrix(&B);
	
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	d_A.stride = A.stride;
	hipMalloc(&d_A.elements, sizeof(int) * size_A);
	hipMemcpy(d_A.elements, A.elements, sizeof(int) * size_A, hipMemcpyHostToDevice);
	
	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	d_B.stride = B.stride;
	hipMalloc(&d_B.elements, sizeof(int) * size_B);
	hipMemcpy(d_B.elements, B.elements, sizeof(int) * size_B, hipMemcpyHostToDevice);
	
	Matrix d_C;
	d_C.width = C.width;
	d_C.height = C.height;
	d_C.stride = C.stride;
	hipMalloc(&d_C.elements, sizeof(int) * size_C);
	
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(C.width / dimBlock.x, C.height / dimBlock.y);
	
	printf("   ------------------------------------------------------------------------------------\n");
	
	hipEvent_t gpustart, gpustop;
	float elapsedTime = 0.0;
	hipEventCreate(&gpustart);
	hipEventCreate(&gpustop);
	hipEventRecord(gpustart, 0);
	hipLaunchKernelGGL(MatMulKernel, dim3(dimGrid), dim3(dimBlock), 0, 0, d_A, d_B, d_C);
	
	hipDeviceSynchronize();
	hipEventRecord(gpustop, 0);
	hipEventSynchronize(gpustop);
	hipEventElapsedTime(&elapsedTime, gpustart, gpustop);
	
	hipMemcpy(C.elements, d_C.elements, sizeof(int) * size_C, hipMemcpyDeviceToHost);
	hipDeviceSynchronize();
	printf("   Matrix_deviceRef: (%d×%d)  <(%d,%d),(%d,%d)>  GPU运行时间为：%f s\n",
		C.height, C.width, dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y, elapsedTime/1000 );
	
	hipEventDestroy(gpustart);
	hipEventDestroy(gpustop);
	
	
	printMatrix(&C);
	
	free(A.elements);
	free(B.elements);
	free(C.elements);
	
	hipFree(d_A.elements);
	hipFree(d_B.elements);
	hipFree(d_C.elements);
	return 0;
}
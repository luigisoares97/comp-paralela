#include <stdio.h>
#include <stdlib.h>

/**
 *  =============== Comparação entre os tempos de execução: ===============
 * Sequencial: 1m18.116s
 * Paralelo: 0m25.027s
 * Paralelo (GPU - OpenMP): 0m15.604s
 * Paralelo (GPU - CUDA): 0m1.534s
 *
 *  =============== Métricas relacionas as versões em GPU  ===============
 * OpenMP:
 *     warps_launched: 535592
 *     warp_execution_efficiency: 98.95%
 *
 * CUDA:
 *     warps_launched: 127008
 *     warp_execution_efficiency: 99.21%
 */
 
__global__ void mm(double* a, double* b, double* c, int width) 
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < width && j < width) {
		double sum = 0;
		
		for (int k = 0; k < width; k++) {
			double x = a[i * width + k];
			double y = b[k * width + j];
			sum += x * y;
		}

		c[i * width + j] = sum;
	}
}

int main()
{
	int width = 2000;
	double *a = (double*) malloc (width * width * sizeof(double));
	double *b = (double*) malloc (width * width * sizeof(double));
	double *c = (double*) malloc (width * width * sizeof(double));

	for(int i = 0; i < width; i++) {
		for(int j = 0; j < width; j++) {
			a[i * width + j] = i;
			b[i * width + j] = j;
			c[i * width + j] = 0;
		}
	}

	int size = width * width * sizeof(double);
	double *cuda_a, *cuda_b, *cuda_c;

	cudaMalloc((void **) &cuda_a, size);
	cudaMemcpy(cuda_a, a, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &cuda_b, size);
	cudaMemcpy(cuda_b, b, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &cuda_c, size);

	int block_size = 32;
	int dim = (width - 1) / block_size + 1;

	dim3 dimGrid(dim, dim, 1);
	dim3 dimBlock(block_size, block_size, 1);

	mm<<<dimGrid, dimBlock>>>(cuda_a, cuda_b, cuda_c, width);

	cudaMemcpy(c, cuda_c, size, cudaMemcpyDeviceToHost);

//	for(int i = 0; i < width; i++) {
//		for(int j = 0; j < width; j++) {
//			printf("\n c[%d][%d] = %f", i, j, c[i * width + j]);
//		}
//	}

	cudaFree(cuda_a);
	cudaFree(cuda_b);
	cudaFree(cuda_c);
}

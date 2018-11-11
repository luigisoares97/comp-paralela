#include <stdio.h>
#include <stdlib.h>

/**
 * Tempo sequencial: 0m0.404s
 * Tempo CUDA: 0m1.970s (overhead de copiar os dados para a GPU)
 */

__global__ void scan_cuda(double* a, double *s, int width) {
	// kernel scan
	int thread = threadIdx.x;
	int block = blockIdx.x * blockDim.x;

	// Cria vetor na memória local.
	__shared__ double partial[1024];

	// Carrega elementos do vetor da memória global para a local.
	if (block + thread < width) {
		partial[thread] = a[block + thread];
	}

	// espera que todas as threads tenham carregado seus elementos
	__syncthreads();

	// Realiza o scan em log n passos.
	for (int i = 1; i < blockDim.x; i *= 2) {
		// Se thread ainda participa, atribui a soma para
		// uma variável temporária.
		double temp = 0;

		if (thread >= i) {
			temp = partial[thread] + partial[thread - i];

			// Espera todas as threads fazerem as somas.
			__syncthreads();

			// Copia o valor calculado em definitivo para o
			// vetor local.
			partial[thread] = temp;
		}

		__syncthreads();
	}

	// Copia da memória local para a global.
	if (block + thread < width) {
		a[block + thread] = partial[thread];
	}

	// Se for a última thread do block, copia o seu valor
	// para o vetor de saída.
	if (thread == blockDim.x - 1) {
		s[blockIdx.x + 1] = a[block + thread];
	}
} 

__global__ void add_cuda(double *a, double *s, int width) {
	// kernel soma
	int thread = threadIdx.x;
	int block = blockIdx.x * blockDim.x;

	// Adiciona o somatório do último elemento do bloco anterior
	// ao elemento atual.
	if (block + thread < width) {
		a[block + thread] += s[blockIdx.x];
	}
}

int main()
{
	int width = 40000000;
	int size = width * sizeof(double);

	int block_size = 1024;
	int num_blocks = (width-1)/block_size + 1;
	int s_size = (num_blocks * sizeof(double));  
 
	double *a = (double*) malloc (size);
	double *s = (double*) malloc (s_size);

	for(int i = 0; i < width; i++)
		a[i] = i;

	double *d_a, *d_s;

	// alocar vetores "a" e "s" no device
	cudaMalloc((void **) &d_a, size);
	cudaMalloc((void **) &d_s, s_size);
	
	// copiar vetor "a" para o device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	
	// definição do número de blocos e threads (dimGrid e dimBlock)
	dim3 dimGrid(num_blocks, 1, 1);
	dim3 dimBlock(block_size, 1, 1);
	
	// chamada do kernel scan
	scan_cuda<<<dimGrid, dimBlock>>>(d_a, d_s, width);
	
	// copiar vetor "s" para o host
	cudaMemcpy(s, d_s, s_size, cudaMemcpyDeviceToHost);
	
	// scan no host (já implementado)
	s[0] = 0;
	for (int i = 1; i < num_blocks; i++)
		s[i] += s[i-1];
 
	// copiar vetor "s" para o device
	cudaMemcpy(d_s, s, s_size, cudaMemcpyHostToDevice);

	// chamada do kernel da soma
	add_cuda<<<dimGrid, dimBlock>>>(d_a, d_s, width);

	// copiar o vetor "a" para o host
	cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);

	printf("\na[%d] = %f\n", width-1, a[width-1]);
  
	cudaFree(d_a);
	cudaFree(d_s);
}

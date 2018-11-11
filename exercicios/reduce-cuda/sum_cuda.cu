#include <stdio.h>
#include <stdlib.h>

/**
 * (1) Tempos de execução:
 *   (a) CUDA: 0m1.626s
 *   (b) Sequencial: 0m0.324s
 *   (c) OpenMP: 0m0.314s
 *   (d) OpenMP - GPU: 0m1.688s
 *   (e) CUDA - Global: 0m1.723s
 *
 * (2) Nvprof:
 *   (a) CUDA:
 *     (a.1) CUDA memcpy HtoD: 465.46ms
 *     (a.2) sum_cuda: 21.560ms
 *   (b) OpenMP - GPU:
 *     (b.1) CUDA memcpy HtoD: 463.25ms
 *      (b.2) sum_cuda: 8.1613ms
 *   (c) CUDA - Global:
 *     (c.1) CUDA memcpy HtoD: 469.75ms
 *     (c.2) sum_cuda: 31.788ms
 *
 * (3) Explicação:
 * Os resultados mostram um tempo de execução maior para aqueles códigos
 * relacionados a GPU. Isso ocorre devido a complexidade do algoritmo (linear).
 * Por ser bastante simples, o custo associado a cópia das informações da memória
 * principal para a memória da GPU não compensa.
 * Em relação ao uso de memória global ou compartilhada (por cada grupo), no CUDA,
 * o que pode-se concluir é que há um prejuízo na utilização da memória global. Isso
 * ocorre, porque o acesso a um dado presenta na memória global é mais custoso. A memória
 * compartilhada é específica para cada grupo de threads, sendo assim o custo de acesso é, logicamente, menor.
 **/

__global__ void sum_cuda(double* a, double *s, int width) {
	int t = threadIdx.x;
	int b = blockIdx.x*blockDim.x;

	// Utilizando memória global
	int i;
	for (i = blockDim.x/2; i > 0; i /= 2) {
		if (t < i && b+t+i < width) {
			a[t + b] += a[t + b + i];
		}
    
		__syncthreads();
	}

	if (t == 0) {
		s[blockIdx.x] = a[t + b];
	}
} 

int main()
{
	int width = 40000000;
	int size = width * sizeof(double);

	int block_size = 1024;
	int num_blocks = (width-1)/block_size+1;
	int s_size = (num_blocks * sizeof(double));  
 
	double *a = (double*) malloc (size);
	double *s = (double*) malloc (s_size);

	for(int i = 0; i < width; i++) {
		a[i] = i;
	}

	double *d_a, *d_s;

	// alocação e cópia dos dados
	cudaMalloc((void **) &d_a, size);
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_s, s_size);

	// definição do número de blocos e threads
	dim3 dimGrid(num_blocks,1,1);
	dim3 dimBlock(block_size,1,1);

	// chamada do kernel
	sum_cuda<<<dimGrid,dimBlock>>>(d_a, d_s, width);

	// cópia dos resultados para o host
	cudaMemcpy(s, d_s, s_size, cudaMemcpyDeviceToHost);

	// soma das reduções parciais
	for(int i = 1; i < num_blocks; i++) 
		s[0] += s[i];

	printf("\nSum = %f\n",s[0]);
  
	cudaFree(d_a);
	cudaFree(d_s);
}

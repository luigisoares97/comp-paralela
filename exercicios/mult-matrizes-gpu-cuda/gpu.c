#include <stdio.h>
#include <stdlib.h>

/**
 *  =============== Comparação entre os tempos de execução: ===============
 * Sequencial: 1m18.116s
 * Paralelo: 0m25.027s
 * Paralelo (GPU - distribute): 4m26.506s
 * Paralelo (GPU - distribute parallel for): 1m25.496s
 * Paralelo (GPU - distribute parallel for simd): 0m15.604s
 *
 *  =============== Métricas relacionas as versões em GPU  ===============
 * Distribute:
 *     warps_launched: 1772568
 *     warp_execution_efficiency: 100%
 *
 * Distribute parallel for:
 *     warps_launched: 814440
 *     warp_execution_efficiency: 100%
 *
 * Distribute parallel for simd:
 *     warps_launched: 535592
 *     warp_execution_efficiency: 98.95%
 */

void mm(double* a, double* b, double* c, int width) 
{
	int m_size = width * width;
	
	#pragma omp target map(to: a[0:m_size], b[0:m_size]) \
		map(from: c[0:m_size])
	#pragma omp teams distribute parallel for collapse(2)
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < width; j++) {
			double sum = 0;

			#pragma omp simd reduction(+: sum)
			for (int k = 0; k < width; k++) {
				double x = a[i * width + k];
				double y = b[k * width + j];
				sum += x * y;
			}
			
			c[i * width + j] = sum;
		}
	}
}

int main()
{
	int width = 2000;
	double *a = (double*) malloc (width * width * sizeof(double));
	double *b = (double*) malloc (width * width * sizeof(double));
	double *c = (double*) malloc (width * width * sizeof(double));

	#pragma omp parallel for collapse(2)
	for(int i = 0; i < width; i++) {
		for(int j = 0; j < width; j++) {
			a[i*width+j] = i;
			b[i*width+j] = j;
			c[i*width+j] = 0;
		}
	}

	mm(a,b,c,width);
	
	for(int i = 0; i < width; i++) {
		for(int j = 0; j < width; j++) {
			printf("\n c[%d][%d] = %f",i,j,c[i*width+j]);
		}
	}
}

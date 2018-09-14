#include <stdio.h>
#include <stdlib.h>

/**
 * Comparação entre tempos de execução sequencial e paralelo:
 * Sequencial: 1m 9,331s
 * Paralelo  : 0m 28,365s
 */

void mm(double* a, double* b, double* c, int width) 
{
	/*
	 * Collapse: cláusula utilizada quando se tem
	 * loops ("imediatamente") aninhandos, fazendo
	 * com que se trate os diferentes níveis de loops
	 * como sendo um só, para distribuir as iterações
	 * as threads.
	 */
	#pragma omp parallel for collapse(2) 
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < width; j++) {
			double sum = 0;

			/*
			 * SIMD (Single-Instruction, Multiple-Data):
			 * diretiva incluida a partir do OpenMP 4.0,
			 * que permite a aplicação de uma única instrução
			 * a múltiplos dados.
			 */
			#pragma omp simd
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
	
	/* for(int i = 0; i < width; i++) { */
		/* for(int j = 0; j < width; j++) { */
			/* printf("\n c[%d][%d] = %f",i,j,c[i*width+j]); */
		/* } */
	/* } */
}

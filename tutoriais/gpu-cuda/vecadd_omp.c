#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void vecadd(double* a, double* b, double* c, int width)
{
	#pragma omp target map(to:a[0:width],b[0:width]) \
		map(tofrom:c[0:width])
	#pragma omp teams distribute parallel for simd
	for (int i = 0; i < width; i++)
		c[i] = a[i] + b[i];
}

int main()
{
	int width = 10000000;
	double *a = (double*) malloc (width * sizeof(double));
	double *b = (double*) malloc (width * sizeof(double));
	double *c = (double*) malloc (width * sizeof(double));

	for(int i = 0; i < width; i++) {
		a[i] = i;
		b[i] = width-i;
		c[i] = 0;
	}

	vecadd(a,b,c,width);

	for(int i = 0; i < width; i++)
		printf("\n c[%d] = %f",i,c[i]);
}

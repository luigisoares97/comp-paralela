#include <stdio.h>
#include <stdlib.h>

double sum_cuda(double* a, int width) {
	double s = 0;

	/* #pragma omp target map(to: a[0:width]) \ */
		/* map(tofrom: s) */
	/* #pragma omp teams distribute parallel for simd reduction(+: s) */
	#pragma omp parallel for simd reduction(+: s)
	for (unsigned i = 0; i < width; i++) {
		s += a[i];
	}

	return s;
}

int main() {
	int width = 40000000;
	double *a = (double*) malloc(sizeof(double) * width);

	for(int i = 0; i < width; i++) {
		a[i] = i;
	}
	
	double s = sum_cuda(a, width);
	printf("\nSum = %f\n", s);	
}

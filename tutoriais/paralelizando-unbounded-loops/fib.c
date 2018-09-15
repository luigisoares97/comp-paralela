#include <stdio.h>

#define N 42

long fib(long n) {
	long i, j;

	if (n < 2) {
		return n;
	} else if (n < 30) {
		return fib(n - 1) + fib(n - 2);
	} else {
		#pragma omp task shared(i)
		i = fib(n - 1);
		
		#pragma omp task shared(j)
		j = fib(n - 2);
		
		#pragma omp taskwait

		return i + j;
	}
}

int main() {

	#pragma omp parallel 
	#pragma omp single
	printf("\nFibonacci(%lu) = %lu\n",(long)N,fib((long)N));
}

#include <stdio.h>

#define N 42

long fib(long n) {
	long i, j;

	if (n < 2) {
		return n;
	}
	else {
		i = fib(n-1);
		j = fib(n-2);

		return i + j;
	}
}

int main() {
	printf("\nFibonacci(%lu) = %lu\n",(long)N,fib((long)N));
}

#include <stdio.h>

#define N 100

int main() {
	int i, j; 
  
	#pragma omp parallel sections private(i, j)
	{
		#pragma omp section
		for(i=0; i < N; i++)
			printf("i");

		#pragma omp section
		for(j=0; j < N; j++)
			printf("j");
	}
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int main()
{
	srand(time(NULL));

	#pragma omp parallel 
	{
		int id = omp_get_thread_num();
		int numero_secreto = rand() % 20; 

		#pragma omp master
		printf("Vamos revelar os números secretos!\n");   

		#pragma omp barrier
		printf("Thread %d escolheu o número %d\n",id,numero_secreto);   

	}
}

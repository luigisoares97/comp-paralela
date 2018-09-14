#include <stdio.h>
#include <omp.h>

int main()
{
	int i;

	// omp parallel: cria uma determinada quantidade de threads.
	#pragma omp parallel num_threads(2) // seta o número de threads em 2 
	{
		int tid = omp_get_thread_num(); // lê o identificador da thread

		/*
		 * omp for: distribui as iteracões do "for" entre as threads.
		 * Vale ressaltar que a combinação "omp parallel" + "omp for" apresenta
		 * o mesmo resultado que o comando "omp parallel for".
		 *
		 * ordered: força que certos eventos dentro do loop sejam executados
		 * em uma ordem definida. Deve existir apenas um bloco "ordered" para
		 * cada loop, sendo que o loop também deve conter a cláusula "ordered".
		 */
		#pragma omp for ordered
		for(i = 1; i <= 3; i++) 
		{
			// Bloco a ser executado em uma ordem específica.
			#pragma omp ordered
			{
				printf("[PRINT1] T%d = %d \n",tid,i);
				printf("[PRINT2] T%d = %d \n",tid,i);
			}
		}
	}
}

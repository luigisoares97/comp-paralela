#include <stdio.h>

int main()
{
	printf("Vamos contar de 1 a 4\n");

	// Padrão map: cada iteração do for é executada
	// somente uma vez, mas cada thread executa uma
	// iteração diferente.
	#pragma omp parallel for
	for (int i = 1; i <= 4; i++)
		printf("%d\n", i);
}

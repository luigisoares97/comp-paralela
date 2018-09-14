#include <stdio.h>

int main()
{
	printf("Vamos contar de 1 a 4\n");

	// Ao declarar o "i" dentro da região paralela (1º for),
	// cada thread passa a ter sua própria cópia privada.
	#pragma omp parallel
	for (int i = 1; i <= 4; i++)
		printf("%d\n", i);

	printf("\n\n");

	// Outra forma de resolver o problema de compartilhamento
	// de uma variável é usar o comando "private".
	int i;

	#pragma omp parallel private(i)
	for (i = 1; i <= 4; i++)
	{
		printf("%d\n", i);
	}
}

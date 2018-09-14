/*
 * Adapted from: http://w...content-available-to-author-only...s.org/sieve-of-eratosthenes
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#ifndef SCHEDULE
#define SCHEDULE schedule(static)
#endif

/**
 * 2ª versão do algoritmo do crivo de eratóstenes paralelo,
 * utilizando o padrão REDUCE e políticas de escalonamento
 * diferentes da política padrão.
 *
 * ==========================================================
 * ======================= Compilação =======================
 * ==========================================================
 *
 * Utilizando a política de escalonamento padrão (static):
 * --> gcc parallel_v2.c -o parallel_v2 -lm -fopenmp
 *
 * Utlizando uma política de escalonamento diferente:
 * --> gcc parallel_v2.c -o parallel_v2 -lm -fopenmp
 * 			-DSCHEDULE='schedule(dynamic, 32)'
 *
 * ==========================================================
 * ======================== Execução ========================
 * ==========================================================
 *
 * time ./parallel_v2
 *
 * ===================== Máquina Local ======================
 * --> sequencial: 0m1,756s
 *
 * (static)
 * --> paralelo: 0m1,392s
 * --> speedup : 1.261
 *
 * (dynamic)
 * --> paralelo: 0m3,332s
 * --> speedup : NÃO HOUVE (chunk size 1 = overhead alto?)
 *
 * (dynamic, 32)
 * --> paralelo: 0m1,439s
 * --> speedup : 1,220 (pior que o estático)
 *
 * (dynamic, 64)
 * --> paralelo: 0m1,398s
 * --> speedup : 1,256 (praticamente igual ao estático)
 *
 * (dynamic, 100)
 * --> paralelo: 0m1,385s
 * --> speedup : 1.267 (praticamente igual ao estático)
 *
 * (dynamic, 128)
 * --> paralelo: 0m1,372s
 * --> speedup : 1.279 (um pouco melhor que os testes acima)
 *
 * (guided)
 * --> paralelo: 0m1,367s
 * --> speedup : 1,284 (melhor caso até então)
 *
 * (guided, 32)
 * --> paralelo: 0m1,366s
 * --> speedup : 1,285 (praticamente igual ao guided padrão)
 *
 * (guided, 64)
 * --> paralelo: 0m1,390s
 * --> speedup : 1,263 (pior em relação aos dois anteriores)
 *
 * (guided, 100)
 * --> paralelo: 0m1,392s
 * --> speedup : 1,261 (praticamente igual ao anterior)
 *
 * (guided, 128)
 * --> paralelo: 0m1,401s
 * --> speedup : 1,253 (pior que os anteriores)
 *
 * Melhor caso: (guided, 32) --> speedup 1,285
 *
 * ======================== PARCODE =========================
 * --> sequencial: 0m4,050s
 *
 * (static)
 * --> paralelo: 0m2,538s
 * --> speedup : 1.595
 *
 * (dynamic)
 * --> paralelo: 0m7,582s
 * --> speedup : NÃO HOUVE (chunk size 1 = overhead alto?)
 *
 * (dynamic, 32)
 * --> paralelo: 0m2,635s
 * --> speedup : 1,537 (pior que o estático)
 *
 * (dynamic, 64)
 * --> paralelo: 0m2,572s
 * --> speedup : 1,574 (pior que o estático, mas melhor que o anterior)
 *
 * (dynamic, 100)
 * --> paralelo: 0m2,560s
 * --> speedup : 1.582 (pior que o estático, mas melhor que o anterior)
 *
 * (dynamic, 128)
 * --> paralelo: 0m2,548s
 * --> speedup : 1.589 (pior que o estático, mas melhor que o anterior)
 *
 * (guided)
 * --> paralelo: 0m2,526s
 * --> speedup : 1,603 (melhor caso até então)
 *
 * (guided, 32)
 * --> paralelo: 0m2,534s
 * --> speedup : 1,598 (pior que o guided padrão, mas bem próximo)
 *
 * (guided, 64)
 * --> paralelo: 0m2,534s
 * --> speedup : 1,598 (igual ao anterior)
 *
 * (guided, 100)
 * --> paralelo: 0m2,533s
 * --> speedup : 1,598 (praticamente igual ao anterior)
 *
 * (guided, 128)
 * --> paralelo: 0m2,540s
 * --> speedup : 1,594 (pior que os anteriores)
 *
 * Melhor caso: (guided) --> speedup 1,603
 *
 */

int sieveOfEratosthenes(int n)
{
	// Create a boolean array "prime[0..n]" and initialize
	// all entries it as true. A value in prime[i] will
	// finally be false if i is Not a prime, else true.
	int primes = 0; 
	bool *prime = (bool*) malloc((n+1)*sizeof(bool));
	int sqrt_n = sqrt(n);
 
	memset(prime, true,(n+1)*sizeof(bool));

	for (int p=2; p <= sqrt_n; p++)
	{
		
		// If prime[p] is not changed, then it is a prime
		if (prime[p] == true)
		{
			/*
			 * Update all multiples of p.
			 * Padrão MAP: cada iteração é executada uma única vez, por alguma
			 * das threads criadas.
			 */
			#pragma omp parallel for num_threads(2)
			for (int i=p*2; i<=n; i += p)
				prime[i] = false;
        }
    }

	/*
     * count prime numbers.
	 *
	 * <prime> é uma variável usada somente para leitura, portanto
	 * não apresenta impactos na paralelização.
	 *
	 * <primes> é uma variável que acumula o resultado referente
	 * a quantidade de números primos, por isso não pode ser privada.
	 *
	 * A quantidade de operações realizadas no loop é fixa e pequena,
	 * inviabilizando a inserção de uma seção crítica, uma vez que as
	 * threads passariam mais tempo em espera do que executando.
	 *
	 * Nessas condições, o mais adequado é aplicar o padrão REDUCE.
	 */
	#pragma omp parallel for SCHEDULE reduction(+: primes) num_threads(2)
	for (int p=2; p<=n; p++)
		if (prime[p])
			primes++;
 
    return(primes);
}
 
int main()
{
	int n = 100000000;
	printf("%d\n",sieveOfEratosthenes(n));
	return 0;
} 

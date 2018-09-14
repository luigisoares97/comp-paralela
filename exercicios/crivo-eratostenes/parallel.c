/*
 * Adapted from: http://w...content-available-to-author-only...s.org/sieve-of-eratosthenes
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

/**
 * Tempo sequencial: 0m 1.704s
 * Tempo paralelo: 0m 1.033s 
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
			#pragma omp parallel for
			for (int i=p*2; i<=n; i += p)
				prime[i] = false;
        }
    }

	/*
     * count prime numbers.
	 * Padrão REDUCE: combina os elementos (0 se não for primo, 1 se for),
	 * reduzindo-os até um único valor.
	 */
	#pragma omp parallel for reduction(+: primes)
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

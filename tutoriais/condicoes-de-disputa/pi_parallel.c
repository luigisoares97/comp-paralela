#include <stdio.h>

/**
 * <soma> depende do valor de <x>.
 * <x> é uma variável compartilhada entre as threads.
 * <x> terá um valor não determinístico, causando resultados
 * incorretos.
 * <x> deve ser uma variável privada.
 *
 * <soma> não pode ser privatizada, porque acumula um resultado
 * compartilhado pelas threads.
 *
 * Variáveis compartilhadas devem ser protegidas por uma seção
 * crítica. Porém, nesse caso, as threads passam a maior parte do
 * tempo esperando a outra na seção crítica, gerando um overhead
 * grande e piorando bastante o tempo de execução.
 *
 * Solução mais adequada: Padrão REDUCE --> cria uma variável
 * privada para cada thread, sendo agrupadas o final.
 */

long long num_passos = 1000000000;
double passo;

int main() {
   int i;
   double x, pi, soma = 0.0;
   passo = 1.0 / (double) num_passos;

   #pragma omp parallel for private(x) reduction(+: soma)
   for (i = 0; i < num_passos; i++) {
      x = (i + 0.5) * passo;
      soma = soma + 4.0 / (1.0 + x * x);
   }

   pi = soma * passo;
	
   printf("O valor de PI é: %f\n", pi);
   return 0;
}

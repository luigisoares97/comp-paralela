#include <stdlib.h>

/**
 * 1) not vectorized: loop contains function calls or data references
 * that cannot be analyzed --> chamada recursiva.
 *
 * 2) not vectorized: control flow in loop --> condicional + break dentro
 * do loop.
 *
 * 3) not vectorized: no grouped stores in basic block --> valores dispostos
 * de forma não contígua na memória.
 *
 * 4) not vectorized: not enough data-refs in basic block --> erro "default".
 *
 * 5) not vectorized: no vectype for stmt: MEM[(float *)vectp_a.41_33] = vect__7.40_27;
 * scalar_type: vector(4) float
 */

float add(float *a, float *b,  int n) {
	float sum = 0;
	
	for (int i = 0; i < n; i += 2) {
		float x = (i + 0.8) * n;
		sum += (a[i] + b[i]) * x;
		
		if (a[i] == b[i] && i > 50) {
			break;
		}
	}

	for (int i = 0; i < n; i++) {
		add(a, b, n);
	}

	return sum;
}

int main() {
	const int N = 100;
	float *a = (float *) malloc(N * sizeof(float));
	float *b = (float *) malloc(N * sizeof(float));
	
	for (int i = 0; i < N; i++) {
		a[i] = (i + 1) / 5.0;
		b[i] = (i + 1) / 7.0;
	}
	
	float sum = add(a, b, N);
}

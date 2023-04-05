#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;

	int chunk;
	#pragma omp parallel
	{
		int nthread = omp_get_num_threads();
		chunk = r / nthread;
		chunk = chunk ? chunk : 1;
	}
	// printf("chunk size = %d\n", chunk); // for debug

	unsigned long long rr = r * r;
	#pragma omp parallel for schedule(guided, chunk) reduction(+ : pixels) firstprivate(rr)
	for (unsigned long long x = 0; x < r; x++) {
		unsigned long long mod_flag = x & 0xffffffULL;
		unsigned long long y = ceil(sqrtl(rr - x*x));
		pixels += y;
		pixels = mod_flag ? pixels : pixels % k;
	}

	printf("%llu\n", (4 * pixels) % k);
}
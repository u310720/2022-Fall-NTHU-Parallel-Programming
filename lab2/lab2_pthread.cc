#include <pthread.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <thread>
#include <omp.h>
// #include <emmintrin.h>

struct calc_pixels_args
{
	unsigned long long r, k;
	unsigned long long begin, end, pixels;
};

void *calc_pixels(void *_args)
{
	calc_pixels_args *args = (calc_pixels_args *)_args;
	for (unsigned long long x = args->begin, rr = args->r * args->r; x < args->end; x++)
	{
		unsigned long long mod_flag = x & 0xffffffULL;
		unsigned long long y = ceil(sqrtl(rr - x * x));
		args->pixels += y;
		args->pixels = mod_flag ? args->pixels : args->pixels % args->k;
	}
	// printf("begin=%llu, end=%llu, pixels=%llu\n", args->begin, args->end, args->pixels);
	pthread_exit(NULL);
	return NULL;
}

int get_num_thread()
{
	int num_threads;
	// printf("num=%d\n", num_threads);
	#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	num_threads *= std::thread::hardware_concurrency();
	// printf("num=%d\n", num_threads);
	return num_threads;
}

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	// cpu_set_t cpuset;
	// sched_getaffinity(0, sizeof(cpuset), &cpuset);
	// unsigned long long ncpus = CPU_COUNT(&cpuset);

	if (r < 10000)
	{
		unsigned long long rr = r * r;
		for (unsigned long long x = 0; x < r; x++)
			pixels += ceil(sqrtl(rr - x*x));
		printf("%llu\n", (4 * pixels) % k);
		return 0;
	}

	int num_threads = get_num_thread();
	pthread_t threads[num_threads];
	calc_pixels_args args[num_threads];
	for (int i = 0, chunk = r / num_threads; i < num_threads; i++)
	{
		args[i].r = r;
		args[i].k = k;
		args[i].begin = chunk * i;
		args[i].end = chunk * (i + 1);
		args[i].pixels = 0;
		// printf("begin=%llu, end=%llu\n", args[i].begin, args[i].end);
	}
	args[num_threads - 1].end = r;

	for (int i = 0, rc; i < num_threads; i++)
	{
		rc = pthread_create(&threads[i], NULL, calc_pixels, (void *)&args[i]);
		if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
	}
	for (int i = 0; i < num_threads; i++)
		if (pthread_join(threads[i], NULL)) {
			printf("error join thread.");
			abort();
		}
	for (int i = 0; i < num_threads; i++)
		pixels += args[i].pixels;	

	printf("%llu\n", (4 * pixels) % k);
	pthread_exit(NULL);
}

#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <thread>

unsigned long long *get_workload(unsigned long long r, unsigned long long n_chunk)
{
	unsigned long long *buf = new unsigned long long[n_chunk * 2](); // init as 0
	const unsigned long long unit_wl = round(r / static_cast<double>(n_chunk)); // workload per process
	unsigned long long l = 0, end = 2 * (n_chunk < r ? n_chunk : r);
	for (unsigned long long idx = 0; idx < end; idx += 2)
	{
		buf[idx] = l;
		buf[idx + 1] = (l += unit_wl);
	}
	buf[end - 1] = r;
	return buf;
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

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0, sum_pixels = 0;
	unsigned long long *range_send;
	unsigned long long range_recv[2] = {0, 0}; // range[begin, end)

	int pid, nproc;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	range_send = pid == 0 ? get_workload(r, nproc) : nullptr;
	MPI_Scatter(range_send, 2, MPI_UNSIGNED_LONG_LONG, &range_recv, 2, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
	
	unsigned long long rr = r * r;
	unsigned long long chunk;
	#pragma omp parallel
	{
		chunk = (range_recv[1] - range_recv[0]) / omp_get_num_threads();
		chunk = chunk ? chunk : 1;
	}
	// printf("pid=%d, begin=%llu, end%llu, chunk=%llu\n", pid, range_recv[0], range_recv[1], chunk);
	#pragma omp parallel for schedule(static, chunk) reduction(+ : pixels) firstprivate(range_recv, rr)
	for (unsigned long long x = range_recv[0]; x < range_recv[1]; x++) {
		unsigned long long mod_flag = x & 0xffffffULL;
		unsigned long long y = ceil(sqrtl(rr - x*x));
		pixels += y;
		pixels = mod_flag ? pixels : pixels % k;
	}

	MPI_Reduce(&pixels, &sum_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	if (pid == 0)
	{
		delete[] range_send;
		printf("%llu\n", (4 * sum_pixels) % k);
	}
	MPI_Finalize();
	return 0;
}

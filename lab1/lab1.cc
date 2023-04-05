#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
typedef unsigned long long __ULL;

#define VERSION 3

#if VERSION == 1
#define BASIC
#elif VERSION == 2
#define BASIC_UNROLL
#define N_UNROLL 32ULL
#elif VERSION == 3
#define DIAGONAL_AND_UNROLL
#define N_UNROLL 32ULL
#define COS45 0.70710678118654752440084436210485L
#endif

#ifdef DEBUG
void Print_Array(__ULL arr[], __ULL sz)
{
	printf("Array={%llu", arr[0]);
	for (__ULL idx = 1; idx < sz; idx++)
		printf(", %llu", arr[idx]);
	printf("}\n");
}
#endif // DEBUG

__ULL *Get_Workload_Buffer(__ULL r, __ULL nproc)
{
	__ULL *buf = new __ULL[nproc * 2](); // init as 0

#if defined(BASIC) || defined(BASIC_UNROLL)
	const __ULL unit_wl = round(r / static_cast<double>(nproc)); // workload per process
	__ULL l = 0, end = 2 * (nproc < r ? nproc : r);
	for (__ULL idx = 0; idx < end; idx += 2)
	{
		buf[idx] = l;
		buf[idx + 1] = (l += unit_wl);
	}
	buf[end - 1] = r;
#endif // BASIC or BASIC_UNROLL

#ifdef DIAGONAL_AND_UNROLL
	const __ULL unit_wl = round(r * COS45 / nproc); // workload per process
	__ULL l = 0, end = 2 * (nproc < r ? nproc : r);
	for (__ULL idx = 0; idx < end; idx += 2)
	{
		buf[idx] = l;
		buf[idx + 1] = (l += unit_wl);
	}
	buf[end - 1] = ceil(r * COS45);
#endif // DIAGONAL_AND_UNROLL

#ifdef DEBUG
	printf("total_workload=%llu, unit_workload=%llu\n", r, unit_wl);
	Print_Array(buf, 2 * nproc);
#endif // DEBUG

	return buf;
}

__ULL Calc_Pixel(const __ULL r, const __ULL mod_k, const __ULL begin, const __ULL end)
{
#ifdef BASIC
	__ULL x(begin), y, pixels(0);
	for (; x < end; x++)
	{
		y = ceil(sqrtl((r + x) * (r - x)));
		pixels += y;
		pixels %= mod_k;
	}
#endif // BASIC

#if defined(BASIC_UNROLL) || defined(DIAGONAL_AND_UNROLL)
	const __ULL unroll_end = ((end - begin) % N_UNROLL == 0) ? end : end - ((end - begin) % N_UNROLL);
	__ULL y[N_UNROLL], pixels(0);
	__ULL x[N_UNROLL];
	for (__ULL i = 0; i < N_UNROLL; i++)
		x[i] = begin + i;
	while (x[0] < unroll_end)
	{
		// loop unrolling
		y[0] = ceil(sqrtl((r + x[0]) * (r - x[0])));
		y[1] = ceil(sqrtl((r + x[1]) * (r - x[1])));
		y[2] = ceil(sqrtl((r + x[2]) * (r - x[2])));
		y[3] = ceil(sqrtl((r + x[3]) * (r - x[3])));
		y[4] = ceil(sqrtl((r + x[4]) * (r - x[4])));
		y[5] = ceil(sqrtl((r + x[5]) * (r - x[5])));
		y[6] = ceil(sqrtl((r + x[6]) * (r - x[6])));
		y[7] = ceil(sqrtl((r + x[7]) * (r - x[7])));
		y[8] = ceil(sqrtl((r + x[8]) * (r - x[8])));
		y[9] = ceil(sqrtl((r + x[9]) * (r - x[9])));
		y[10] = ceil(sqrtl((r + x[10]) * (r - x[10])));
		y[11] = ceil(sqrtl((r + x[11]) * (r - x[11])));
		y[12] = ceil(sqrtl((r + x[12]) * (r - x[12])));
		y[13] = ceil(sqrtl((r + x[13]) * (r - x[13])));
		y[14] = ceil(sqrtl((r + x[14]) * (r - x[14])));
		y[15] = ceil(sqrtl((r + x[15]) * (r - x[15])));
		y[16] = ceil(sqrtl((r + x[16]) * (r - x[16])));
		y[17] = ceil(sqrtl((r + x[17]) * (r - x[17])));
		y[18] = ceil(sqrtl((r + x[18]) * (r - x[18])));
		y[19] = ceil(sqrtl((r + x[19]) * (r - x[19])));
		y[20] = ceil(sqrtl((r + x[20]) * (r - x[20])));
		y[21] = ceil(sqrtl((r + x[21]) * (r - x[21])));
		y[22] = ceil(sqrtl((r + x[22]) * (r - x[22])));
		y[23] = ceil(sqrtl((r + x[23]) * (r - x[23])));
		y[24] = ceil(sqrtl((r + x[24]) * (r - x[24])));
		y[25] = ceil(sqrtl((r + x[25]) * (r - x[25])));
		y[26] = ceil(sqrtl((r + x[26]) * (r - x[26])));
		y[27] = ceil(sqrtl((r + x[27]) * (r - x[27])));
		y[28] = ceil(sqrtl((r + x[28]) * (r - x[28])));
		y[29] = ceil(sqrtl((r + x[29]) * (r - x[29])));
		y[30] = ceil(sqrtl((r + x[30]) * (r - x[30])));
		y[31] = ceil(sqrtl((r + x[31]) * (r - x[31])));

#ifdef DIAGONAL_AND_UNROLL
		y[0] -= x[0];
		y[1] -= x[1];
		y[2] -= x[2];
		y[3] -= x[3];
		y[4] -= x[4];
		y[5] -= x[5];
		y[6] -= x[6];
		y[7] -= x[7];
		y[8] -= x[8];
		y[9] -= x[9];
		y[10] -= x[10];
		y[11] -= x[11];
		y[12] -= x[12];
		y[13] -= x[13];
		y[14] -= x[14];
		y[15] -= x[15];
		y[16] -= x[16];
		y[17] -= x[17];
		y[18] -= x[18];
		y[19] -= x[19];
		y[20] -= x[20];
		y[21] -= x[21];
		y[22] -= x[22];
		y[23] -= x[23];
		y[24] -= x[24];
		y[25] -= x[25];
		y[26] -= x[26];
		y[27] -= x[27];
		y[28] -= x[28];
		y[29] -= x[29];
		y[30] -= x[30];
		y[31] -= x[31];
#endif // DIAGONAL_AND_UNROLL

		// sum the result
		y[0] += y[1];
		y[2] += y[3];
		y[4] += y[5];
		y[6] += y[7];
		y[8] += y[9];
		y[10] += y[11];
		y[12] += y[13];
		y[14] += y[15];
		y[16] += y[17];
		y[18] += y[19];
		y[20] += y[21];
		y[22] += y[23];
		y[24] += y[25];
		y[26] += y[27];
		y[28] += y[29];
		y[30] += y[31];
		y[0] += y[2];
		y[4] += y[6];
		y[8] += y[10];
		y[12] += y[14];
		y[16] += y[18];
		y[20] += y[22];
		y[24] += y[26];
		y[28] += y[30];
		y[0] += y[4];
		y[8] += y[12];
		y[16] += y[20];
		y[24] += y[28];
		y[0] += y[8];
		y[16] += y[24];

#ifdef DIAGONAL_AND_UNROLL
		pixels -= N_UNROLL;
#endif

		pixels += y[0] + y[16];
		pixels %= mod_k;

		// next round
		x[0] += N_UNROLL;
		x[1] += N_UNROLL;
		x[2] += N_UNROLL;
		x[3] += N_UNROLL;
		x[4] += N_UNROLL;
		x[5] += N_UNROLL;
		x[6] += N_UNROLL;
		x[7] += N_UNROLL;
		x[8] += N_UNROLL;
		x[9] += N_UNROLL;
		x[10] += N_UNROLL;
		x[11] += N_UNROLL;
		x[12] += N_UNROLL;
		x[13] += N_UNROLL;
		x[14] += N_UNROLL;
		x[15] += N_UNROLL;
		x[16] += N_UNROLL;
		x[17] += N_UNROLL;
		x[18] += N_UNROLL;
		x[19] += N_UNROLL;
		x[20] += N_UNROLL;
		x[21] += N_UNROLL;
		x[22] += N_UNROLL;
		x[23] += N_UNROLL;
		x[24] += N_UNROLL;
		x[25] += N_UNROLL;
		x[26] += N_UNROLL;
		x[27] += N_UNROLL;
		x[28] += N_UNROLL;
		x[29] += N_UNROLL;
		x[30] += N_UNROLL;
		x[31] += N_UNROLL;
	}
	for (; x[0] < end; ++x[0])
	{
		y[0] = ceil(sqrtl((r + x[0]) * (r - (x[0]))));

#ifdef DIAGONAL_AND_UNROLL
		y[0] -= x[0] + 1;
#endif // DIAGONAL_AND_UNROLL

		pixels += y[0];
		pixels %= mod_k;
	}

#ifdef DEBUG
	printf("unroll_end=%llu, end=%llu\n", unroll_end, end);
#endif // DEBUG

#endif // BASIC_UNROLL or DIAGONAL_AND_UNROLL

	return pixels;
}

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

	MPI_Init(&argc, &argv);

	const __ULL root = 0;
	const __ULL r = atoll(argv[1]);
	const __ULL k = atoll(argv[2]);

	int rank = -1;
	__ULL quarter_pixels;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (r < 100) // no message passing
	{
		if (rank == root)
		{
#if defined(BASIC) || defined(BASIC_UNROLL)
			quarter_pixels = Calc_Pixel(r, k, 0, r);
#endif // BASIC or BASIC_UNROLL

#ifdef DIAGONAL_AND_UNROLL
			__ULL diag_pixels = ceil(r * COS45);
			quarter_pixels = Calc_Pixel(r, k, 0, diag_pixels);
			quarter_pixels *= 2;
			quarter_pixels += diag_pixels;
#endif // DIAGONAL_AND_UNROLL
		}
	}
	else // message passing
	{
		int nproc; // due to the definition of MPI, nproc must be int
		MPI_Comm_size(MPI_COMM_WORLD, &nproc);

		// set the calculation range for each process
		__ULL range[2] = {0, 0}; // range[begin, end)
		__ULL *range_buf = rank == root ? Get_Workload_Buffer(r, nproc) : nullptr;
		MPI_Scatter(range_buf, 2, MPI_UNSIGNED_LONG_LONG, &range, 2, MPI_UNSIGNED_LONG_LONG, root, MPI_COMM_WORLD);
		if (rank == root)
			delete[] range_buf;

		const __ULL local_pixels = Calc_Pixel(r, k, range[0], range[1]);
		
#ifdef DEBUG
		printf("Rank=%llu, Range(%llu, %llu)\n", rank, range[0], range[1]);
		printf("Rank=%llu local_pixels=%llu\n", rank, local_pixels);
#endif // DEBUG

		MPI_Reduce(&local_pixels, &quarter_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, root, MPI_COMM_WORLD);

#ifdef DIAGONAL_AND_UNROLL
		// add the pixels on the other side of the diagonal
		quarter_pixels += quarter_pixels;

		// add diagonal pixels
		quarter_pixels += ceil(r * COS45);
#endif // DIAGONAL_AND_UNROLL
	}

	if (rank == root)
		printf("%llu\n", (4 * quarter_pixels) % k);

	MPI_Finalize();
	return 0;
}
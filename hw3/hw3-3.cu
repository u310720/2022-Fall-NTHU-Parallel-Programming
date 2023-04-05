#include <stdio.h>
#include <stdlib.h>

constexpr int ceil(int a, int b) { return (a + b - 1) / b; }
static const int INF = ((1 << 30) - 1);
static const int BLK_WIDTH = 2;

struct HostData
{
    int nV, nE;
    int nPadV; // padding elements to fit the width of the block
    int *H_Dist = NULL;

    HostData(int nVertex, int nEdge) : nV(nVertex), nE(nEdge)
    {
        nPadV = ceil(nV, BLK_WIDTH) * BLK_WIDTH;
        cudaMallocHost(&H_Dist, nPadV * nPadV * sizeof(int));
    }
    ~HostData()
    {
        cudaFree(H_Dist);
    }
};

/* util */
inline int h_index(int i, int j, int row_size) { return i * row_size + j; }
__device__ inline int d_index(int i, int j, int row_size) { return i * row_size + j; }

/* debug */
void h_printMatrix(int *arr, int width);
__device__ void d_printMatrix(int *arr, int width);

/* IO */
HostData *input(char *inFileName);
void output(char *outFileName, const HostData *hData);

/* APSP */
void blk_FW(int nV, int nPadV, int *H_Dist);
__global__ void naiveCudaFWKernal(int *D_Dist, int k, int nPadV);
__global__ void naiveCudaFW(int *D_Dist, int nV, int nPadV);
__global__ void calPhase1(int *D_Dist, int pivot, int nPadV);
__global__ void calPhase2(int *D_Dist, int pivot, int nPadV, int devID);
__global__ void calPhase3(int *D_Dist, int pivot, int nPadV, int devID);

int main(int argc, char *argv[])
{
    /* input */
    HostData *hData = input(argv[1]);
    printf("%d\n", hData->nPadV);

    /* blocked Floyd-Washall */
    // h_printMatrix(H_Dist, nPadV); // debug
    blk_FW(hData->nV, hData->nPadV, hData->H_Dist);

    /* naive Floyd-Washall */
    // dim3 dimGrid(nPadV / BLK_WIDTH, nPadV / BLK_WIDTH);
    // dim3 dimBlock(BLK_WIDTH, BLK_WIDTH);
    // cudaFuncSetCacheConfig(naiveCudaFWKernal, cudaFuncCachePreferL1);
    // for (int k = 0; k < hData->nV; ++k)
    //     naiveCudaFWKernal <<< dimGrid, dimBlock >>> (D_Dist, k, nPadV);

    /* output */
    // h_printMatrix(H_Dist, nPadV); // debug
    output(argv[2], hData);
    delete hData;

    return 0;
}

/* blocked FW */
void blk_FW(int nV, int nPadV, int *H_Dist)
{
    const int nBlk = ceil(nV, BLK_WIDTH);
    dim3 dimGridPhase1(1, 1);
    dim3 dimGridPhase2(nPadV / BLK_WIDTH, 2); // blockIdx.y is the flag that marks it as a column or a row.
    dim3 dimGridPhase3(nPadV / BLK_WIDTH, nPadV / BLK_WIDTH);
    dim3 dimBlk(BLK_WIDTH, BLK_WIDTH);

    int *D0_Dist = NULL;
    int *D1_Dist = NULL;
    cudaSetDevice(0);
    cudaHostGetDevicePointer(&D0_Dist, H_Dist, 0);
    cudaSetDevice(1);
    cudaHostGetDevicePointer(&D1_Dist, H_Dist, 0);

    // h_printMatrix(H_Dist, nPadV); // for debug
    for (int pivot = 0; pivot < nBlk; ++pivot)
    {
        /* phase 1 */
        calPhase1<<<dimGridPhase1, dimBlk>>>(D0_Dist, pivot, nPadV);
        cudaDeviceSynchronize();

        /* phase 2 */
        cudaSetDevice(0);
        calPhase2<<<dimGridPhase2, dimBlk>>>(D0_Dist, pivot, nPadV, 0);
        cudaSetDevice(1);
        calPhase2<<<dimGridPhase2, dimBlk>>>(D1_Dist, pivot, nPadV, 1);
        cudaSetDevice(0);
        cudaDeviceSynchronize();
        cudaSetDevice(1);
        cudaDeviceSynchronize();

        /* phase 3 */
        cudaSetDevice(0);
        calPhase3<<<dimGridPhase3, dimBlk>>>(D0_Dist, pivot, nPadV, 0);
        cudaSetDevice(1);
        calPhase3<<<dimGridPhase3, dimBlk>>>(D1_Dist, pivot, nPadV, 1);
        cudaSetDevice(0);
        cudaDeviceSynchronize();
        cudaSetDevice(1);
        cudaDeviceSynchronize();
    }
    // h_printMatrix(H_Dist, nPadV); // for debug
}
__global__ void calPhase1(int *D_Dist, int pivot, int nPadV)
{
    const int idx = threadIdx.x;            // share memory index
    const int idy = threadIdx.y;            // share memory index
    const int v1 = pivot * BLK_WIDTH + idy; // global memory index
    const int v2 = pivot * BLK_WIDTH + idx; // global memory index

    /* load */
    const int s_index = d_index(idy, idx, BLK_WIDTH);
    // const int g_index = d_index(v1, v2, nPadV);
    __shared__ int S_Dist_Blk_Update[BLK_WIDTH * BLK_WIDTH];
    S_Dist_Blk_Update[s_index] = D_Dist[d_index(v1, v2, nPadV)];
    __syncthreads();

/* debug, check data in share memory */
// if (idx == 0 && idy == 0)
// {
//     printf("blkIdx.x=%d, blkIdx.y=%d\n", blockIdx.x, blockIdx.y);
//     for (int i = 0; i < BLK_WIDTH; ++i)
//         for (int j = 0; j < BLK_WIDTH; ++j)
//             printf("S_Dist_Blk_Update[%d][%d]=%d\n", idx + i, idy + j, S_Dist_Blk_Update[d_index(idx + i, idy + j, BLK_WIDTH)]);
// }

/* computing */
#pragma unroll
    for (int k = 0; k < BLK_WIDTH; ++k)
    {
        const int new_dist = S_Dist_Blk_Update[d_index(idy, k, BLK_WIDTH)] + S_Dist_Blk_Update[d_index(k, idx, BLK_WIDTH)];
        if (new_dist < S_Dist_Blk_Update[s_index])
            S_Dist_Blk_Update[s_index] = new_dist;
        __syncthreads();
    }

    /* store */
    D_Dist[d_index(v1, v2, nPadV)] = S_Dist_Blk_Update[s_index];
}
__global__ void calPhase2(int *D_Dist, int pivot, int nPadV, int devID)
{
    /* exception */
    if ((blockIdx.x == pivot) || // phase 1
        (blockIdx.y != devID)    // 2-devices
    )
        return;

    /* variables */
    const int idx = threadIdx.x;            // share memory index
    const int idy = threadIdx.y;            // share memory index
    const int v1 = pivot * BLK_WIDTH + idy; // global memory index
    const int v2 = pivot * BLK_WIDTH + idx; // global memory index

    /* load */
    const int s_index = d_index(idy, idx, BLK_WIDTH);
    const int g_index = (blockIdx.y == 0) ? d_index(v1, blockIdx.x * BLK_WIDTH + idx, nPadV) : d_index(blockIdx.x * BLK_WIDTH + idy, v2, nPadV);
    __shared__ int S_Dist_Blk_Base[BLK_WIDTH * BLK_WIDTH];
    __shared__ int S_Dist_Blk_Update[BLK_WIDTH * BLK_WIDTH];
    S_Dist_Blk_Base[s_index] = D_Dist[d_index(v1, v2, nPadV)];
    S_Dist_Blk_Update[s_index] = D_Dist[g_index];
    __syncthreads();

    /* debug, check data in share memory */
    // if (idx == 0 && idy == 0 && blockIdx.x == 1 && blockIdx.y == 0)
    // {
    //     for (int i = 0; i < BLK_WIDTH; ++i)
    //         for (int j = 0; j < BLK_WIDTH; ++j)
    //             printf("blkIdx.x=%d, blkIdx.y=%d, S_Dist_Blk_Base[%d][%d]=%d\n", blockIdx.x, blockIdx.y, idx + i, idy + j, S_Dist_Blk_Base[d_index(idx + i, idy + j, BLK_WIDTH)]);
    //     for (int i = 0; i < BLK_WIDTH; ++i)
    //         for (int j = 0; j < BLK_WIDTH; ++j)
    //             printf("blkIdx.x=%d, blkIdx.y=%d, S_Dist_Blk_Update[%d][%d]=%d\n", blockIdx.x, blockIdx.y, idx + i, idy + j, S_Dist_Blk_Update[d_index(idx + i, idy + j, BLK_WIDTH)]);
    // }

    /* computing */
    if (blockIdx.y == 0)
    {
#pragma unroll
        for (int k = 0; k < BLK_WIDTH; ++k)
        {
            const int new_dist = S_Dist_Blk_Base[d_index(idy, k, BLK_WIDTH)] + S_Dist_Blk_Update[d_index(k, idx, BLK_WIDTH)];
            if (new_dist < S_Dist_Blk_Update[s_index])
                S_Dist_Blk_Update[s_index] = new_dist;
            __syncthreads();
        }
    }
    else
    {
#pragma unroll
        for (int k = 0; k < BLK_WIDTH; ++k)
        {
            // const int new_dist = S_Dist_Blk_Update[d_index(idy, k, BLK_WIDTH)] + S_Dist_Blk_Base[d_index(k, idx, BLK_WIDTH)];
            if (S_Dist_Blk_Update[d_index(idy, k, BLK_WIDTH)] + S_Dist_Blk_Base[d_index(k, idx, BLK_WIDTH)] < S_Dist_Blk_Update[s_index])
                S_Dist_Blk_Update[s_index] = S_Dist_Blk_Update[d_index(idy, k, BLK_WIDTH)] + S_Dist_Blk_Base[d_index(k, idx, BLK_WIDTH)];
            __syncthreads();
        }
    }

    /* store */
    D_Dist[g_index] = S_Dist_Blk_Update[s_index];
}
__global__ void calPhase3(int *D_Dist, int pivot, int nPadV, int devID)
{
    /* exception */
    if ((blockIdx.x == pivot || blockIdx.y == pivot) ||                        // phase 1, 2
        (devID == 0 && blockIdx.x > pivot || devID == 1 && blockIdx.x < pivot) // 2 devices
    )
        return;

    /* variables */
    const int idx = threadIdx.x;                 // share memory index
    const int idy = threadIdx.y;                 // share memory index
    const int v1 = blockIdx.y * BLK_WIDTH + idy; // global memory index
    const int v2 = blockIdx.x * BLK_WIDTH + idx; // global memory index
    const int pv1 = pivot * BLK_WIDTH + idy;     // global memory index
    const int pv2 = pivot * BLK_WIDTH + idx;     // global memory index

    /* load */
    const int s_index = d_index(idy, idx, BLK_WIDTH);
    // const int g_index = d_index(v1, v2, nPadV);
    int min_dist = D_Dist[d_index(v1, v2, nPadV)];
    __shared__ int S_Dist_Blk_Row[BLK_WIDTH * BLK_WIDTH];
    __shared__ int S_Dist_Blk_Col[BLK_WIDTH * BLK_WIDTH];
    S_Dist_Blk_Row[s_index] = D_Dist[d_index(pv1, v2, nPadV)];
    S_Dist_Blk_Col[s_index] = D_Dist[d_index(v1, pv2, nPadV)];
    __syncthreads();

/* debug, check data in share memory */
// if (idx == 0 && idy == 0 && blockIdx.x == 1 && blockIdx.y == 2)
// {
//     for (int i = 0; i < BLK_WIDTH; ++i)
//         for (int j = 0; j < BLK_WIDTH; ++j)
//             printf("blkIdx.x=%d, blkIdx.y=%d, S_Dist_Blk_Row[%d][%d]=%d\n", blockIdx.x, blockIdx.y, idx + i, idy + j, S_Dist_Blk_Row[d_index(idx + i, idy + j, BLK_WIDTH)]);
//     for (int i = 0; i < BLK_WIDTH; ++i)
//         for (int j = 0; j < BLK_WIDTH; ++j)
//             printf("blkIdx.x=%d, blkIdx.y=%d, S_Dist_Blk_Col[%d][%d]=%d\n", blockIdx.x, blockIdx.y, idx + i, idy + j, S_Dist_Blk_Col[d_index(idx + i, idy + j, BLK_WIDTH)]);
// }

/* computing */
#pragma unroll
    for (int k = 0; k < BLK_WIDTH; ++k)
    {
        const int new_dist = S_Dist_Blk_Col[d_index(idy, k, BLK_WIDTH)] + S_Dist_Blk_Row[d_index(k, idx, BLK_WIDTH)];
        if (new_dist < min_dist)
            min_dist = new_dist;
        // __syncthreads(); // no data dependency here
    }

    /* store */
    D_Dist[d_index(v1, v2, nPadV)] = min_dist;
}

/* naive FW */
__global__ void naiveCudaFWKernal(int *D_Dist, int k, int nPadV)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int new_dist = D_Dist[d_index(i, k, nPadV)] + D_Dist[d_index(k, j, nPadV)];

    if (new_dist < D_Dist[i * nPadV + j])
        D_Dist[i * nPadV + j] = new_dist;
}
__global__ void naiveCudaFW(int *D_Dist, int nV, int nPadV) // result is not correct
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;

    for (int k = 0; k < nV; ++k)
    {
        const int new_dist = D_Dist[d_index(i, k, nPadV)] + D_Dist[d_index(k, j, nPadV)];
        if (new_dist < D_Dist[i * nPadV + j])
            D_Dist[i * nPadV + j] = new_dist;
        __syncthreads();
    }
}

/* IO */
HostData *input(char *infile)
{
    FILE *file = fopen(infile, "rb");
    if (!file)
    {
        printf("Could not open %s\n", infile);
        fflush(stdout);
        exit(EXIT_FAILURE);
    }

    int nV, nE;
    size_t vf = fread(&nV, sizeof(int), 1, file);
    size_t ef = fread(&nE, sizeof(int), 1, file);
    HostData *hData(NULL);
    hData = new HostData(nV, nE);

    for (int i = 0; i < hData->nPadV; ++i)
    {
        for (int j = 0; j < hData->nPadV; ++j)
        {
            if (i == j && i < nV && j < nV)
            {
                (hData->H_Dist)[h_index(i, j, hData->nPadV)] = 0;
            }
            else
            {
                (hData->H_Dist)[h_index(i, j, hData->nPadV)] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < nE; ++i)
    {
        size_t pf = fread(pair, sizeof(int), 3, file);
        (hData->H_Dist)[h_index(pair[0], pair[1], hData->nPadV)] = pair[2];
    }
    fclose(file);

    return hData;
}
void output(char *outFileName, const HostData *hData)
{
    FILE *outfile = fopen(outFileName, "w");
    for (int i = 0; i < hData->nV; ++i)
    {
        for (int j = 0; j < hData->nV; ++j)
        {
            if ((hData->H_Dist)[h_index(i, j, hData->nPadV)] >= INF)
                (hData->H_Dist)[h_index(i, j, hData->nPadV)] = INF;
        }
        fwrite(&(hData->H_Dist)[h_index(i, 0, hData->nPadV)], sizeof(int), hData->nV, outfile);
    }
    fclose(outfile);
}

/* debug */
void h_printMatrix(int *arr, int width)
{
    printf("------------------------------------------------------------------------------\n");
    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < width; ++y)
            printf("%12d\t", arr[x * width + y]);
        printf("\n");
    }
    printf("------------------------------------------------------------------------------\n");
}
__device__ void d_printMatrix(int *arr, int width)
{
    printf("------------------------------------------------------------------------------\n");
    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < width; ++y)
            printf("%12d\t", arr[x * width + y]);
        printf("\n");
    }
    printf("------------------------------------------------------------------------------\n");
}
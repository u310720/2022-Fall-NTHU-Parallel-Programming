// #include <cassert>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>
using namespace std;

/* for johnson algo. */
// #include <queue>
// #include <vector>
// #include <list>
// #include <algorithm>

static const int INF((1 << 30) - 1);

// #define __INTRINSIC__
#ifdef __INTRINSIC__
#include <nmmintrin.h> // sse4.2
static const __m128i INF_128(_mm_set1_epi32(INF));
#endif // __INTRINSIC__

class Graph
{
public:
    // ctor
    Graph(const char *file);
    ~Graph() { free(dist); }

    /* getter */
    int num_vertex() const { return nV; }
    int num_edge() const { return nE; }
    int d(int src, int dst) const { return dist[src * row_size + dst]; }
    int d(int idx) const { return dist[idx]; }

    /* setter */
    void set_dist(int src, int dst, int val) { dist[src * row_size + dst] = val; }
    void set_dist(int idx, int val) { dist[idx] = val; }

    // APSP
    void floyd_warshall();
    // void johnson();
    // void SPFA(); // no impl
    void output_all_pairs_shortest_path(const char *file);

    // debug
    void print_dist();
    void print_ans(const char *);

private:
    int nV;       // number of vertices
    int nE;       // number of edges
    int row_size; //
    // int *weight;  // adjacency matrix
    int *dist; // a matrix storing point-to-point distances

    /* for johnson algo. */
    // typedef list<pair<int, int>> Adjlist;
    // vector<Adjlist> adjlist;
    // void dijkstra(int src);
};

int main(int argc, char **argv)
{
    // assert(argc == 3);
    Graph graph(argv[1]);
    graph.floyd_warshall();
    // graph.johnson();

    /* for debug */
    // cout << "------------------------------------------------------------------\n";
    // graph.print_weights();
    // cout << "------------------------------------------------------------------\n";
    // graph.print_dist();
    // cout << "------------------------------------------------------------------\n";
    // graph.print_ans(string("./cases/" + string(argv[2])).c_str());
    // cout << "------------------------------------------------------------------\n";

    graph.output_all_pairs_shortest_path(argv[2]);
    return 0;
}

// input format (binary file whose contents are 32-bit integers):
// nV nE src(1) dst(1) weight(1) src(2) dst(2) weight(2) ...
// constraints:
// 2 <= nV <= 6000 (CPU)
// 2 <= nV <= 40000 (Single-GPU)
// 2 <= nV <= 60000 (Multi-GPU)
// 0 <= nE <= V*(V−1)
// 0 <= src(i), dst(i) < nV
// src(i) != dst(i)
// if src(i) == src(j) then dst(i) != dst(j) (there will not be repeated edges)
// 0 <= weight(i) <= 1000
// distance(src, dst) == 2^30 - 1 when there is no valid path from src to dst
Graph::Graph(const char *file) : nV(0), nE(0), row_size(0), dist(nullptr)
{
    ifstream ifs(file, ios::in | ios::binary);
    if (!ifs)
    {
        cerr << "Cannot open " << file << endl;
        exit(EXIT_FAILURE);
    }

    ifs.read(reinterpret_cast<char *>(&nV), sizeof(int));
    ifs.read(reinterpret_cast<char *>(&nE), sizeof(int));
#ifdef __INTRINSIC__
    row_size = (nV & 0x03) ? ((nV >> 2) << 2) + 4 : nV;
#else
    row_size = nV;
#endif // __INTRINSIC__

    dist = (int *)malloc(nV * row_size * sizeof(int));

#pragma omp parallel for schedule(static) default(shared)
    for (int i = 0; i < nV; ++i)
        for (int j = 0; j < nV; ++j)
            if (i == j)
                set_dist(i, j, 0);
            else
                set_dist(i, j, INF);

    const int nE_3 = nE * 3;
    int *edges(nullptr);
    edges = (int *)malloc(nE_3 * sizeof(int));
    ifs.read(reinterpret_cast<char *>(edges), nE_3 * sizeof(int));
#pragma omp parallel for schedule(static) default(shared)
    for (int i = 0; i < nE_3; i += 3)
        set_dist(edges[i], edges[i + 1], edges[i + 2]);
    free(edges);
}
void Graph::floyd_warshall()
{
    if (nV < 4)
    {
        // algorithm:
        for (int k = 0; k < nV; ++k)
            for (int src = 0; src < nV; ++src)
                for (int dst = 0; dst < nV; ++dst)
                    if (d(src, k) + d(k, dst) < d(src, dst))
                        set_dist(src, dst, d(src, k) + d(k, dst));
        return;
    }

#ifdef __INTRINSIC__
    for (int k = 0; k < nV; ++k)
    {
        const int *const d_k = &dist[k * row_size];
#pragma omp parallel for schedule(static) default(shared)
        for (int src = 0; src < nV; ++src)
        {
            int *const d_src = &dist[src * row_size];
            const __m128i d_src_k = _mm_set1_epi32(d_src[k]);

            for (int dst = 0; dst < row_size; dst += 4)
            {
                // __m128i d_src_dst = _mm_load_si128((__m128i const *)&d_src[dst]);
                // __m128i d_k_dst = _mm_load_si128((__m128i const *)&d_k[dst]);

                _mm_store_si128(
                    (__m128i *)&d_src[dst],
                    _mm_min_epi32(
                        _mm_load_si128((__m128i const *)&d_src[dst]),
                        _mm_add_epi32(
                            d_src_k,
                            _mm_load_si128((__m128i const *)&d_k[dst]))));
            }
        }
    }
#else
    const int __nV = (nV >> 2) << 2;
    for (int k = 0; k < nV; ++k)
    {
        const int *const d_k = &dist[k * row_size];
#pragma omp parallel for schedule(static) default(shared)
        for (int src = 0; src < nV; ++src)
        {
            int *const d_src = &dist[src * row_size];
            int dst;

            for (dst = 0; dst < __nV; dst += 4)
            {
                if (d_src[k] + d_k[dst] < d_src[dst])
                    d_src[dst] = d_src[k] + d_k[dst];
                if (d_src[k] + d_k[dst + 1] < d_src[dst + 1])
                    d_src[dst + 1] = d_src[k] + d_k[dst + 1];
                if (d_src[k] + d_k[dst + 2] < d_src[dst + 2])
                    d_src[dst + 2] = d_src[k] + d_k[dst + 2];
                if (d_src[k] + d_k[dst + 3] < d_src[dst + 3])
                    d_src[dst + 3] = d_src[k] + d_k[dst + 3];
            }
            for (dst = __nV; dst < nV; ++dst)
                if (d_src[k] + d_k[dst] < d_src[dst])
                    d_src[dst] = d_src[k] + d_k[dst];
        }
        // print_dist(); // for debug
    }
#endif // __INTRINSIC__
}
/* void Graph::dijkstra(int src)
{
    using Node = pair<int, int>;
    int *const d = dist + src * nV;
    vector<int> finish(nV, 0);

    // init distance
    for (int i = 0; i < nV; ++i)
        d[i] = INF;
    d[src] = 0;

    // init minheap
    vector<Node> minheap(nV);
    auto dist_greater = [](const Node &a, const Node &b) -> bool
    { return a.second > b.second; };
    for (int i = 0; i < nV; ++i)
    {
        minheap[i].first = i;
        minheap[i].second = d[i];
    }
    make_heap(minheap.begin(), minheap.end(), dist_greater);

    // algo
    while (!minheap.empty())
    {
        int cur = minheap.front().first;
        pop_heap(minheap.begin(), minheap.end(), dist_greater);
        minheap.pop_back();

        if (finish[cur])
            continue;

        for (auto node : adjlist[cur])
        {
            const int &next = node.first;
            const int &weight = node.second;
            // relax
            if (d[next] > d[cur] + weight)
                d[next] = d[cur] + weight;

            // decrease key
            minheap.push_back(make_pair(next, d[next]));
            push_heap(minheap.begin(), minheap.end(), dist_greater);
        }
        finish[cur] = true;
    }
} */
/* void Graph::johnson()
{
    // Since the testcases do not contain negatively weighted edges,
    // there is no need to re-weight edges

    adjlist.resize(nV);
#pragma omp parallel for schedule(static) default(shared) collapse(2)
    for (int src = 0; src < nV; ++src)
        for (int dst = 0; dst < nV; ++dst)
            if (w(src, dst) != INF)
                adjlist[src].push_back(make_pair(dst, w(src, dst)));

#pragma omp parallel for schedule(static) default(shared)
    for (int src = 0; src < nV; ++src)
        dijkstra(src);
} */
void Graph::output_all_pairs_shortest_path(const char *file)
{
    ofstream ofs(file, ios::out | ios::binary);
    if (!ofs)
    {
        cerr << "Cannot open " << file << endl;
        exit(EXIT_FAILURE);
    }

#ifdef __INTRINSIC__
    for (int i = 0; i < nV; ++i)
        ofs.write(reinterpret_cast<char *>(&dist[i * row_size]), nV * sizeof(int));
#else
    ofs.write(reinterpret_cast<char *>(dist), nV * nV * sizeof(int));
#endif // __INTRINSIC__
}
void Graph::print_dist() // for debug
{
    for (int src = 0; src < nV; src++)
    {
        for (int dst = 0; dst < nV; dst++)
            cout << setw(12) << left << d(src, dst);
        cout << "\n";
    }
}
void Graph::print_ans(const char *file) // for debug
{
    ifstream ifs(file);
    if (!ifs)
    {
        cerr << "Cannot open " << file << endl;
        exit(EXIT_FAILURE);
    }
    const int size = nV * nV;
    int *ans = new int[size];
    ifs.read(reinterpret_cast<char *>(ans), size * sizeof(int));
    for (int i = 0; i < size; ++i)
    {
        cout << setw(12) << left << ans[i];
        if (i % nV == nV - 1)
            cout << endl;
    }
    delete[] ans;
}
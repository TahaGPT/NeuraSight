// Name: Triangle Counting - V4 (Hybrid: OpenMP Threads + AVX2 Intrinsics)
// Assignment: 2 - Parallel Triangle Counting using SIMD and OpenMP
// Version: V4 – Hybrid SIMD intrinsics + OpenMP threading
//
// Description:
//   Combines the two forms of parallelism:
//   • OpenMP parallel threads handle the outer vertex loop (V3 strategy).
//   • AVX2 SIMD intrinsics accelerate each per-edge intersection (V1 kernel).
//
//   This is the recommended "best-effort" configuration:
//   – Dynamic scheduling (chunk 16) handles load imbalance.
//   – 8-lane AVX2 all-pairs intersection replaces scalar merge.
//   – Thread-private no state is needed; each intersection is independent.
//
// Compilation requires: -fopenmp -mavx2 -O3

#include <algorithm>
#include <chrono>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Graph – flat CSR (same as V3)
// ─────────────────────────────────────────────────────────────────────────────

struct Graph {
    int n;
    long long m;
    std::vector<int> row_ptr;
    std::vector<int> col;
};

static Graph load_graph(const std::string& path) {
    std::ifstream fin(path);
    if (!fin.is_open()) { std::cerr << "ERROR: cannot open " << path << "\n"; std::exit(1); }

    std::vector<std::pair<int,int>> raw_edges;
    int max_id = 0;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#' || line[0] == '%') continue;
        std::istringstream iss(line);
        int u, v;
        if (!(iss >> u >> v)) continue;
        if (u == v) continue;
        if (u > v) std::swap(u, v);
        raw_edges.push_back({u, v});
        max_id = std::max(max_id, std::max(u, v));
    }
    std::sort(raw_edges.begin(), raw_edges.end());
    raw_edges.erase(std::unique(raw_edges.begin(), raw_edges.end()), raw_edges.end());

    int N = max_id + 1;
    std::vector<int> deg(N, 0);
    for (auto& e : raw_edges) { deg[e.first]++; deg[e.second]++; }

    std::vector<int> order(N);
    for (int i = 0; i < N; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return deg[a] < deg[b] || (deg[a] == deg[b] && a < b);
    });
    std::vector<int> new_id(N);
    for (int i = 0; i < N; i++) new_id[order[i]] = i;

    std::vector<std::vector<int>> tmp(N);
    for (auto& e : raw_edges) {
        int u = new_id[e.first], v = new_id[e.second];
        if (u > v) std::swap(u, v);
        tmp[u].push_back(v);
    }
    for (int i = 0; i < N; i++) std::sort(tmp[i].begin(), tmp[i].end());

    std::vector<int> row_ptr(N + 1, 0);
    for (int i = 0; i < N; i++) row_ptr[i + 1] = row_ptr[i] + (int)tmp[i].size();
    long long M = row_ptr[N];
    std::vector<int> col(M);
    for (int i = 0; i < N; i++)
        for (int k = 0; k < (int)tmp[i].size(); k++)
            col[row_ptr[i] + k] = tmp[i][k];

    Graph g; g.n = N; g.m = M; g.row_ptr = std::move(row_ptr); g.col = std::move(col);
    return g;
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar tail
// ─────────────────────────────────────────────────────────────────────────────

static inline long long scalar_intersect(const int* a, int na, const int* b, int nb) {
    long long cnt = 0;
    int i = 0, j = 0;
    while (i < na && j < nb) {
        if      (a[i] < b[j]) ++i;
        else if (a[i] > b[j]) ++j;
        else { ++cnt; ++i; ++j; }
    }
    return cnt;
}

// ─────────────────────────────────────────────────────────────────────────────
// AVX2 SIMD kernel  (same as V1)
// ─────────────────────────────────────────────────────────────────────────────

static inline long long avx2_intersect(const int* a, int na, const int* b, int nb) {
    long long cnt = 0;
    const int LANES = 8;

    static const int perm_arr[8] = {1,2,3,4,5,6,7,0};
    __m256i perm = _mm256_loadu_si256((__m256i*)perm_arr);

    int i = 0, j = 0;
    while (i + LANES <= na && j + LANES <= nb) {
        __m256i va = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i*)(b + j));

        __m256i sum = _mm256_setzero_si256();
        for (int r = 0; r < LANES; r++) {
            __m256i cmp = _mm256_cmpeq_epi32(va, vb);
            sum = _mm256_sub_epi32(sum, cmp);
            vb = _mm256_permutevar8x32_epi32(vb, perm);
        }
        __m128i lo = _mm256_castsi256_si128(sum);
        __m128i hi = _mm256_extracti128_si256(sum, 1);
        __m128i s  = _mm_add_epi32(lo, hi);
        s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
        s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
        cnt += (long long)_mm_cvtsi128_si32(s);

        if      (a[i + LANES - 1] < b[j + LANES - 1]) i += LANES;
        else if (a[i + LANES - 1] > b[j + LANES - 1]) j += LANES;
        else { i += LANES; j += LANES; }
    }
    cnt += scalar_intersect(a + i, na - i, b + j, nb - j);
    return cnt;
}

// ─────────────────────────────────────────────────────────────────────────────
// Hybrid triangle counting: OpenMP outer + AVX2 inner
// ─────────────────────────────────────────────────────────────────────────────

long long count_triangles(const Graph& g, int nthreads) {
    omp_set_num_threads(nthreads);
    const int* rp  = g.row_ptr.data();
    const int* col = g.col.data();
    long long tc = 0;

    #pragma omp parallel for schedule(dynamic, 16) reduction(+:tc)
    for (int vi = 0; vi < g.n; vi++) {
        int ai_start = rp[vi], ai_end = rp[vi + 1];
        if (ai_start == ai_end) continue;
        const int* ai = col + ai_start;
        int na_full   = ai_end - ai_start;

        for (int ki = 0; ki < na_full; ki++) {
            int vj = ai[ki];
            int aj_start = rp[vj], aj_end = rp[vj + 1];
            if (aj_start == aj_end) continue;
            int nb = aj_end - aj_start;
            int offset = ki + 1;
            int na = na_full - offset;
            if (na <= 0) continue;
            tc += avx2_intersect(ai + offset, na, col + aj_start, nb);
        }
    }
    return tc;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> <baseline_time>\n";
        return 1;
    }
    double baseline_time = std::stod(argv[2]);
    Graph g = load_graph(argv[1]);

    int nthreads = omp_get_max_threads();

    auto t0 = std::chrono::high_resolution_clock::now();
    long long triangles = count_triangles(g, nthreads);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double speedup  = baseline_time / elapsed;

    std::cout << "=== V4: Hybrid – AVX2 Intrinsics + OpenMP Threads ===\n";
    std::cout << "Threads used   : " << nthreads     << "\n";
    std::cout << "Vertices       : " << g.n          << "\n";
    std::cout << "Edges          : " << g.m          << "\n";
    std::cout << "Triangles      : " << triangles    << "\n";
    std::cout << "Time (s)       : " << elapsed      << "\n";
    std::cout << "Speedup        : " << speedup      << "x\n";
    std::cout << "RESULT v4 " << elapsed << " " << triangles << "\n";
    return 0;
}

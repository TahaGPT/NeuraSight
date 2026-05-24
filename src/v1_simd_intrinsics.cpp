// Name: Triangle Counting - V1 (SIMD Intrinsics Only)
// Assignment: 2 - Parallel Triangle Counting using SIMD and OpenMP
// Version: V1 – SIMD-only using AVX2 intrinsics
//
// Description:
//   Accelerates the sorted-list intersection using AVX2 SIMD intrinsics.
//   The method loads 8 int32 elements at a time from each list and performs
//   an all-pairs comparison via shuffles (similar to the SSE approach in
//   Zhang et al. HPEC 2018, extended to AVX2 8-lane registers).
//
//   Algorithm:
//     • Same degree-ordered, upper-triangular graph as baseline.
//     • Inner intersection kernel replaced with AVX2 vectorised version:
//       - Compare 8-element blocks from list A against 8-element blocks from B.
//       - Advance the pointer of the list whose last element is smaller.
//       - Fall back to scalar for the tail when remaining elements < 8.
//
// Compilation requires: -mavx2

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <immintrin.h>   // AVX2
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Graph loading  (identical to baseline)
// ─────────────────────────────────────────────────────────────────────────────

struct Graph {
    int n;
    long long m;
    std::vector<std::vector<int>> adj;
};

Graph load_graph(const std::string& path) {
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

    std::vector<std::vector<int>> adj(N);
    for (auto& e : raw_edges) {
        int u = new_id[e.first], v = new_id[e.second];
        if (u > v) std::swap(u, v);
        adj[u].push_back(v);
    }
    for (int i = 0; i < N; i++) std::sort(adj[i].begin(), adj[i].end());

    long long total_edges = 0;
    for (int i = 0; i < N; i++) total_edges += adj[i].size();

    Graph g; g.n = N; g.m = total_edges; g.adj = std::move(adj);
    return g;
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar tail helper
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
// AVX2 SIMD intersection kernel
// ─────────────────────────────────────────────────────────────────────────────
//
// Strategy (from Zhang et al. HPEC 2018, extended to 8-wide AVX2):
//   Load a block of 8 ints from A and B.
//   Perform all-to-all comparisons by rotating one register through 8 positions
//   using _mm256_permutevar8x32_epi32, accumulating matches with _mm256_cmpeq_epi32.
//   Advance the pointer whose last element is smaller.

static inline long long avx2_intersect(const int* a, int na, const int* b, int nb) {
    long long cnt = 0;
    const int LANES = 8; // AVX2: 256-bit / 32-bit = 8 ints

    // Rotation permutation indices for cycling B block left by 1..7
    static const int perm_arr[8] = {1,2,3,4,5,6,7,0};
    __m256i perm = _mm256_loadu_si256((__m256i*)perm_arr);

    int i = 0, j = 0;
    while (i + LANES <= na && j + LANES <= nb) {
        __m256i va = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i*)(b + j));

        // All-to-all comparison: rotate vb through 8 positions
        __m256i sum = _mm256_setzero_si256();
        for (int r = 0; r < LANES; r++) {
            __m256i cmp = _mm256_cmpeq_epi32(va, vb);
            // Each matching lane gives 0xFFFFFFFF = -1, so subtract to add 1
            sum = _mm256_sub_epi32(sum, cmp);
            vb = _mm256_permutevar8x32_epi32(vb, perm);
        }
        // Horizontal sum of 8 x int32 lanes
        __m128i lo = _mm256_castsi256_si128(sum);
        __m128i hi = _mm256_extracti128_si256(sum, 1);
        __m128i s  = _mm_add_epi32(lo, hi);
        s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
        s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
        cnt += (long long)_mm_cvtsi128_si32(s);

        // Advance the list whose last element is smaller
        if (a[i + LANES - 1] < b[j + LANES - 1])      i += LANES;
        else if (a[i + LANES - 1] > b[j + LANES - 1]) j += LANES;
        else { i += LANES; j += LANES; }
    }

    // Scalar tail
    cnt += scalar_intersect(a + i, na - i, b + j, nb - j);
    return cnt;
}

// ─────────────────────────────────────────────────────────────────────────────
// Triangle counting
// ─────────────────────────────────────────────────────────────────────────────

long long count_triangles(const Graph& g) {
    long long tc = 0;
    for (int vi = 0; vi < g.n; vi++) {
        const auto& ai = g.adj[vi];
        if (ai.empty()) continue;
        for (int vj : ai) {
            const auto& aj = g.adj[vj];
            if (aj.empty()) continue;
            auto it = std::lower_bound(ai.begin(), ai.end(), vj + 1);
            int offset = (int)(it - ai.begin());
            int na = (int)ai.size() - offset;
            int nb = (int)aj.size();
            if (na <= 0 || nb <= 0) continue;
            tc += avx2_intersect(ai.data() + offset, na, aj.data(), nb);
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

    auto t0 = std::chrono::high_resolution_clock::now();
    long long triangles = count_triangles(g);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double speedup  = baseline_time / elapsed;

    std::cout << "=== V1: SIMD Intrinsics (AVX2) Only ===\n";
    std::cout << "Vertices       : " << g.n          << "\n";
    std::cout << "Edges          : " << g.m          << "\n";
    std::cout << "Triangles      : " << triangles    << "\n";
    std::cout << "Time (s)       : " << elapsed      << "\n";
    std::cout << "Speedup        : " << speedup      << "x\n";
    std::cout << "RESULT v1 " << elapsed << " " << triangles << "\n";
    return 0;
}

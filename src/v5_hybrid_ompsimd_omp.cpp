// Name: Triangle Counting - V5 (Hybrid: OpenMP SIMD + OpenMP Threading)
// Assignment: 2 - Parallel Triangle Counting using SIMD and OpenMP
// Version: V5 – Hybrid OpenMP SIMD directives + OpenMP thread parallelism
//
// Description:
//   Combines both OpenMP features without hand-written intrinsics:
//   • #pragma omp parallel for (outer) – distributes vertices across threads.
//   • #pragma omp simd             (inner) – vectorises the intersection scan.
//
//   The intersection uses a window-based approach so the SIMD clause can
//   apply to a contiguous, branch-free inner loop:
//     For each element x in the shorter list, binary-search to find its
//     candidate window in the longer list, then use #pragma omp simd
//     reduction(+:cnt) over that window.
//
//   This version is fully portable – the compiler emits the best SIMD
//   instructions available (AVX2 on i5-10310U with -mavx2).
//
// Compilation requires: -fopenmp -mavx2 -O3

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Graph – flat CSR
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
// OpenMP SIMD intersection
// ─────────────────────────────────────────────────────────────────────────────
//
// For each element in the shorter list A, we probe the longer list B:
//   1. Binary-search in B for the position of a[i].
//   2. Run a small #pragma omp simd loop over a window [lo, lo+WIN) in B.
//      The window size WIN = 8 matches the AVX2 lane count so the compiler
//      generates a single 256-bit load + compare.
// Correctness: the binary search guarantees we only examine the segment
// where b[k] == a[i] is possible (b is sorted, so exactly one match at most).

static const int WIN = 8;   // AVX2 lane count for int32

static inline long long omp_simd_intersect(const int* __restrict__ a, int na,
                                           const int* __restrict__ b, int nb) {
    // Ensure a is the shorter list
    if (na > nb) { std::swap(a, b); std::swap(na, nb); }

    long long total = 0;
    int jstart = 0;  // b pointer – monotonically advances (merge-order)

    for (int i = 0; i < na; i++) {
        int key = a[i];

        // Advance jstart to first b[j] >= key
        while (jstart < nb && b[jstart] < key) jstart++;
        if (jstart >= nb) break;

        // Scan window [jstart, jstart+WIN) with SIMD
        int jend = jstart + WIN;
        if (jend > nb) jend = nb;

        int local = 0;
        const int* bp = b + jstart;
        int wlen = jend - jstart;

        #pragma omp simd reduction(+:local)
        for (int k = 0; k < wlen; k++) {
            local += (bp[k] == key) ? 1 : 0;
        }
        total += local;

        // If b[jstart] == key, both pointers advance
        if (b[jstart] == key) jstart++;
    }
    return total;
}

// ─────────────────────────────────────────────────────────────────────────────
// Hybrid triangle counting
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
            tc += omp_simd_intersect(ai + offset, na, col + aj_start, nb);
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

    std::cout << "=== V5: Hybrid – OpenMP SIMD + OpenMP Threading ===\n";
    std::cout << "Threads used   : " << nthreads     << "\n";
    std::cout << "Vertices       : " << g.n          << "\n";
    std::cout << "Edges          : " << g.m          << "\n";
    std::cout << "Triangles      : " << triangles    << "\n";
    std::cout << "Time (s)       : " << elapsed      << "\n";
    std::cout << "Speedup        : " << speedup      << "x\n";
    std::cout << "RESULT v5 " << elapsed << " " << triangles << "\n";
    return 0;
}

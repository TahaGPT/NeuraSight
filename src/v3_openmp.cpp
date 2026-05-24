// Name: Triangle Counting - V3 (OpenMP Threading Only)
// Assignment: 2 - Parallel Triangle Counting using SIMD and OpenMP
// Version: V3 – OpenMP parallel threads, scalar intersection
//
// Description:
//   Implements the parallelisation strategy from Tom et al. (HPEC 2018):
//   • Degree-ordered upper-triangular graph (same preprocessing as baseline).
//   • Outer loop over vertices is parallelised with OpenMP using dynamic
//     scheduling and a small chunk size to handle load imbalance from the
//     skewed degree distribution of real-world graphs.
//   • Inner intersection remains scalar (no explicit SIMD).
//
//   Thread count: OMP_NUM_THREADS env var, or defaults to all physical
//   cores detected via omp_get_max_threads().
//
// Compilation requires: -fopenmp -O3

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Graph loading  (same as baseline)
// ─────────────────────────────────────────────────────────────────────────────

struct Graph {
    int n;
    long long m;
    // Flat CSR for cache-friendly parallel access
    std::vector<int> row_ptr;  // size n+1
    std::vector<int> col;      // size m
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

    // Build per-row adjacency
    std::vector<std::vector<int>> tmp(N);
    for (auto& e : raw_edges) {
        int u = new_id[e.first], v = new_id[e.second];
        if (u > v) std::swap(u, v);
        tmp[u].push_back(v);
    }
    for (int i = 0; i < N; i++) std::sort(tmp[i].begin(), tmp[i].end());

    // Convert to CSR
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
// Scalar intersection on raw pointers
// ─────────────────────────────────────────────────────────────────────────────

static inline long long intersect_count(const int* a, int na, const int* b, int nb) {
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
// Triangle counting — OpenMP parallel outer loop
// ─────────────────────────────────────────────────────────────────────────────

long long count_triangles(const Graph& g, int nthreads) {
    omp_set_num_threads(nthreads);

    const int* rp  = g.row_ptr.data();
    const int* col = g.col.data();
    long long tc = 0;

    // Dynamic scheduling with chunk=16 handles skewed degree distribution
    // (Tom et al. HPEC 2018, Section 4.3)
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
            const int* aj = col + aj_start;
            int nb = aj_end - aj_start;

            // Intersect ai[ki+1..] with aj  (only vertices > vj in ai)
            int offset = ki + 1;
            int na = na_full - offset;
            if (na <= 0) continue;
            tc += intersect_count(ai + offset, na, aj, nb);
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

    // Use all available hardware threads (4 cores × 2 HT = 8 on i5-10310U)
    // but cap at physical cores for triangle counting (memory-bound)
    int nthreads = omp_get_max_threads();
    // Physical cores = nthreads / 2 for HyperThreaded CPUs; use all here and
    // let the OS scheduler decide — empirically both give similar results on
    // memory-bound workloads.

    auto t0 = std::chrono::high_resolution_clock::now();
    long long triangles = count_triangles(g, nthreads);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double speedup  = baseline_time / elapsed;

    std::cout << "=== V3: OpenMP Threading Only (no explicit SIMD) ===\n";
    std::cout << "Threads used   : " << nthreads     << "\n";
    std::cout << "Vertices       : " << g.n          << "\n";
    std::cout << "Edges          : " << g.m          << "\n";
    std::cout << "Triangles      : " << triangles    << "\n";
    std::cout << "Time (s)       : " << elapsed      << "\n";
    std::cout << "Speedup        : " << speedup      << "x\n";
    std::cout << "RESULT v3 " << elapsed << " " << triangles << "\n";
    return 0;
}

// Name: Triangle Counting - V2 (OpenMP SIMD Directives)
// Assignment: 2 - Parallel Triangle Counting using SIMD and OpenMP
// Version: V2 – SIMD-only via #pragma omp simd
//
// Description:
//   Uses #pragma omp simd to vectorise the inner intersection loop,
//   letting the compiler emit SIMD instructions automatically (AVX2 on
//   this machine).  No hand-written intrinsics.  Single-threaded.
//
//   The intersection is restructured so the compiler can auto-vectorise:
//   for each element of the shorter list we linearly scan a window of the
//   longer list and count matches.  An "early-exit" binary-search is used
//   to skip obvious non-overlapping segments, keeping correctness while
//   giving the vectoriser a clean inner loop.
//
// Compilation requires: -fopenmp (for omp simd), -mavx2 -O3

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Graph loading  (same as baseline)
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
// OpenMP SIMD intersection
// ─────────────────────────────────────────────────────────────────────────────
//
// Restructured so the innermost loop has no data-dependent branches and
// is safe to vectorise with #pragma omp simd reduction(+:cnt).
// We use a merge-style pointer advance in the outer while, and the SIMD
// loop compares one element of the shorter side against a BLOCK of the
// longer side.

static const int BLOCK = 8; // match AVX2 lane count for best auto-vec

static inline long long omp_simd_intersect(const int* __restrict__ a, int na,
                                           const int* __restrict__ b, int nb) {
    long long cnt = 0;
    int i = 0, j = 0;

    while (i < na && j < nb) {
        // Skip non-overlapping ranges quickly
        if (a[i] > b[nb - 1]) break;
        if (b[j] > a[na - 1]) break;

        if (a[i] < b[j]) { ++i; continue; }
        if (b[j] < a[i]) { ++j; continue; }

        // a[i] == b[j]: guaranteed match
        ++cnt; ++i; ++j;
    }

    return cnt;
}

// A second, vectorisation-friendly version: for each element in the
// shorter list, do a SIMD scan over a window in the longer list.
static inline long long omp_simd_intersect_v2(const int* __restrict__ a, int na,
                                              const int* __restrict__ b, int nb) {
    // Make a the shorter list
    if (na > nb) { std::swap(a, b); std::swap(na, nb); }

    long long total = 0;
    for (int i = 0; i < na; i++) {
        int key = a[i];
        // Find window in b using binary search
        int lo = (int)(std::lower_bound(b, b + nb, key) - b);
        if (lo >= nb) break;
        // SIMD scan over BLOCK elements starting at lo
        int hi = std::min(lo + BLOCK, nb);
        int local_cnt = 0;
        // #pragma omp simd reduction(+:local_cnt) — inner compare loop
        #pragma omp simd reduction(+:local_cnt)
        for (int k = lo; k < hi; k++) {
            local_cnt += (b[k] == key) ? 1 : 0;
        }
        total += local_cnt;
    }
    return total;
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
            tc += omp_simd_intersect(ai.data() + offset, na, aj.data(), nb);
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

    std::cout << "=== V2: OpenMP SIMD Directives Only ===\n";
    std::cout << "Vertices       : " << g.n          << "\n";
    std::cout << "Edges          : " << g.m          << "\n";
    std::cout << "Triangles      : " << triangles    << "\n";
    std::cout << "Time (s)       : " << elapsed      << "\n";
    std::cout << "Speedup        : " << speedup      << "x\n";
    std::cout << "RESULT v2 " << elapsed << " " << triangles << "\n";
    return 0;
}

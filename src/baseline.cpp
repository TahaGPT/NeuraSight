// Name: Triangle Counting - Baseline (Scalar Sequential)
// Assignment: 2 - Parallel Triangle Counting using SIMD and OpenMP
// Version: Baseline
//
// Description:
//   Scalar sequential triangle counting using degree-ordered adjacency lists
//   and sorted-merge intersection. Implements the compact-forward algorithm:
//   edges are directed from low-degree to high-degree vertex, then triangles
//   are counted via sorted-list intersection (OrderedMerge style).
//
// Reference: Tom et al. (HPEC 2018), Shun & Tangwongsan (ICDE 2015)

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Graph loading and preprocessing
// ─────────────────────────────────────────────────────────────────────────────

struct Graph {
    int n;                          // number of vertices
    long long m;                    // number of edges (directed, upper-triangular)
    std::vector<std::vector<int>> adj; // adjacency list (directed: low-deg → high-deg)
};

// Load an undirected edge-list file.
// Removes self-loops and duplicate edges, then applies degree-based ordering.
Graph load_graph(const std::string& path) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "ERROR: cannot open " << path << "\n";
        std::exit(1);
    }

    // Pass 1: collect all edges, detect max vertex id
    std::vector<std::pair<int,int>> raw_edges;
    int max_id = 0;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#' || line[0] == '%') continue;
        std::istringstream iss(line);
        int u, v;
        if (!(iss >> u >> v)) continue;
        if (u == v) continue;          // skip self-loops
        if (u > v) std::swap(u, v);   // canonical form
        raw_edges.push_back({u, v});
        max_id = std::max(max_id, std::max(u, v));
    }

    // Remove duplicate edges
    std::sort(raw_edges.begin(), raw_edges.end());
    raw_edges.erase(std::unique(raw_edges.begin(), raw_edges.end()), raw_edges.end());

    int N = max_id + 1;

    // Compute degrees
    std::vector<int> deg(N, 0);
    for (auto& e : raw_edges) {
        deg[e.first]++;
        deg[e.second]++;
    }

    // Build degree-based vertex ordering (non-decreasing degree → new id)
    // Vertices with smaller degree get smaller new ids.
    // Ties broken by original id for determinism.
    std::vector<int> order(N);
    for (int i = 0; i < N; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return deg[a] < deg[b] || (deg[a] == deg[b] && a < b);
    });

    std::vector<int> new_id(N);
    for (int i = 0; i < N; i++) new_id[order[i]] = i;

    // Build directed adjacency lists (edge goes from lower new_id to higher new_id)
    std::vector<std::vector<int>> adj(N);
    for (auto& e : raw_edges) {
        int u = new_id[e.first];
        int v = new_id[e.second];
        if (u > v) std::swap(u, v);
        adj[u].push_back(v);
    }
    for (int i = 0; i < N; i++)
        std::sort(adj[i].begin(), adj[i].end());

    long long total_edges = 0;
    for (int i = 0; i < N; i++) total_edges += adj[i].size();

    Graph g;
    g.n   = N;
    g.m   = total_edges;
    g.adj = std::move(adj);
    return g;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sorted-list intersection count  (scalar)
// ─────────────────────────────────────────────────────────────────────────────

// Count |A ∩ B| where both arrays are sorted in ascending order.
inline long long intersect_count(const int* a, int na, const int* b, int nb) {
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
// Triangle counting  (hvi, vj, vki ordering — upper triangular CSR)
// ─────────────────────────────────────────────────────────────────────────────

long long count_triangles(const Graph& g) {
    long long tc = 0;
    for (int vi = 0; vi < g.n; vi++) {
        const auto& ai = g.adj[vi];
        if (ai.empty()) continue;
        for (int vj : ai) {
            const auto& aj = g.adj[vj];
            if (aj.empty()) continue;
            // Intersect adj[vi] (entries > vj) with adj[vj]
            // Find start position in ai beyond vj
            auto it = std::lower_bound(ai.begin(), ai.end(), vj + 1);
            int offset = (int)(it - ai.begin());
            tc += intersect_count(
                ai.data() + offset, (int)ai.size() - offset,
                aj.data(), (int)aj.size()
            );
        }
    }
    return tc;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph_file>\n";
        return 1;
    }

    Graph g = load_graph(argv[1]);

    // Count undirected edges (original)
    long long undirected_edges = g.m; // already directed (each undirected edge stored once)

    auto t0 = std::chrono::high_resolution_clock::now();
    long long triangles = count_triangles(g);
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "=== Baseline (Scalar Sequential) ===\n";
    std::cout << "Vertices       : " << g.n          << "\n";
    std::cout << "Edges          : " << undirected_edges << "\n";
    std::cout << "Triangles      : " << triangles    << "\n";
    std::cout << "Time (s)       : " << elapsed      << "\n";
    std::cout << "Speedup        : 1.00x (reference)\n";

    // Machine-readable line for Makefile parsing
    std::cout << "RESULT baseline " << elapsed << " " << triangles << "\n";
    return 0;
}

# ─────────────────────────────────────────────────────────────────────────────
# Makefile – Triangle Counting Benchmark Suite
# Assignment 2: Parallel Triangle Counting using SIMD and OpenMP
#
# Target machine: Intel Core i5-10310U (Comet Lake)
#   • 4 physical cores, 8 hardware threads (HyperThreading)
#   • AVX2 support (256-bit SIMD, 8 × int32 lanes)
#   • L1d: 128 KiB, L2: 1 MiB, L3: 6 MiB
#
# Usage:
#   make                           – build all binaries
#   make run                       – benchmark on all default datasets
#   make run GRAPHS="a.txt b.txt"  – benchmark on custom dataset list
#   make clean                     – remove binaries and result files
# ─────────────────────────────────────────────────────────────────────────────

CXX      := g++
CXXFLAGS := -std=c++17 -O3 -mavx2 -march=native -fopenmp -Wall -Wextra

BINARIES := baseline v1 v2 v3 v4 v5

# Space-separated list of graph files to benchmark
GRAPHS ?= email-Eu-core.txt com-lj.ungraph.txt facebook_combined.txt

# Aggregated results file (fully rewritten each run)
RESULTS_FILE := results_summary.txt

.PHONY: all run clean hw_info _print_summary

# ─────────────────────────────────────────────────────────────────────────────
# Build targets
# ─────────────────────────────────────────────────────────────────────────────

all: $(BINARIES)

baseline: baseline.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

v1: v1_simd_intrinsics.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

v2: v2_omp_simd.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

v3: v3_openmp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

v4: v4_hybrid_intrinsics_omp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

v5: v5_hybrid_ompsimd_omp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# ─────────────────────────────────────────────────────────────────────────────
# Hardware capability discovery
# ─────────────────────────────────────────────────────────────────────────────

hw_info:
	@echo "============================================================"
	@echo " Hardware Capability Discovery"
	@echo "============================================================"
	@echo ""
	@echo "--- CPU Summary ---"
	@lscpu | grep -E "Model name|Architecture|CPU\(s\):|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|CPU max MHz|L[123][di]:"
	@echo ""
	@echo "--- SIMD Support ---"
	@grep -m1 "flags" /proc/cpuinfo | tr ' ' '\n' | \
	 grep -E "^(avx|avx2|avx512|sse4_2|fma)$$" | sort | uniq | \
	 awk '{printf "  %-15s detected\n", $$1}'
	@echo ""
	@NCORES=$$(lscpu | grep "Core(s) per socket" | awk '{print $$NF}'); \
	 NSOCKET=$$(lscpu | grep "Socket(s)"          | awk '{print $$NF}'); \
	 NHT=$$(lscpu    | grep "Thread(s) per core"  | awk '{print $$NF}'); \
	 TOTAL=$$((NCORES * NSOCKET * NHT)); \
	 echo "--- Threading ---"; \
	 echo "  Physical cores   : $$((NCORES * NSOCKET))"; \
	 echo "  Hardware threads : $$TOTAL"; \
	 echo "  OMP threads used : $$TOTAL (all HW threads)"; \
	 echo ""
	@echo "--- SIMD Register Width ---"
	@if grep -q "avx512" /proc/cpuinfo; then \
		echo "  Width: 512 bits — 16 x int32 lanes (AVX-512)"; \
	elif grep -q "avx2" /proc/cpuinfo; then \
		echo "  Width: 256 bits — 8 x int32 lanes (AVX2)  [ACTIVE]"; \
	elif grep -q "avx" /proc/cpuinfo; then \
		echo "  Width: 256 bits — 8 x float lanes (AVX, integer limited)"; \
	elif grep -q "sse4_2" /proc/cpuinfo; then \
		echo "  Width: 128 bits — 4 x int32 lanes (SSE4.2)"; \
	fi
	@echo ""
	@echo "--- Thread Count Justification ---"
	@echo "  i5-10310U has 4 physical cores with Hyper-Threading (2 threads/core)."
	@echo "  Triangle counting is memory-bound for large graphs; both physical"
	@echo "  and HT threads are used (8 total) to maximise L3-cache utilisation."
	@echo "  Dynamic scheduling (chunk=16) handles skewed degree distributions."
	@echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Master benchmark runner – iterates over all graphs in $(GRAPHS)
# ─────────────────────────────────────────────────────────────────────────────

run: all hw_info
	@rm -f $(RESULTS_FILE)
	@echo "============================================================"  >> $(RESULTS_FILE)
	@echo " FULL BENCHMARK SUMMARY"                                        >> $(RESULTS_FILE)
	@echo "============================================================"  >> $(RESULTS_FILE)
	@echo ""                                                               >> $(RESULTS_FILE)
	@for GRAPH in $(GRAPHS); do \
		if [ ! -f "$$GRAPH" ]; then \
			echo ""; \
			echo "SKIP: $$GRAPH not found — place it in this directory."; \
			continue; \
		fi; \
		echo ""; \
		echo "============================================================"; \
		echo " Triangle Counting Benchmark"; \
		echo " Graph: $$GRAPH"; \
		echo "============================================================"; \
		echo ""; \
		echo "--- Running Baseline (scalar sequential) ---"; \
		./baseline $$GRAPH | tee .baseline_out.txt; \
		BT=$$(grep "^RESULT baseline" .baseline_out.txt | awk '{print $$3}'); \
		echo $$BT > .baseline_time.txt; \
		echo ""; \
		echo "--- Running V1: SIMD Intrinsics (AVX2) only ---"; \
		./v1 $$GRAPH $$BT | tee .v1_out.txt; echo ""; \
		echo "--- Running V2: OpenMP SIMD directives only ---"; \
		./v2 $$GRAPH $$BT | tee .v2_out.txt; echo ""; \
		echo "--- Running V3: OpenMP threading (scalar intersection) ---"; \
		./v3 $$GRAPH $$BT | tee .v3_out.txt; echo ""; \
		echo "--- Running V4: Hybrid AVX2 intrinsics + OpenMP threads ---"; \
		./v4 $$GRAPH $$BT | tee .v4_out.txt; echo ""; \
		echo "--- Running V5: Hybrid OpenMP SIMD + OpenMP threads ---"; \
		./v5 $$GRAPH $$BT | tee .v5_out.txt; echo ""; \
		$(MAKE) -s _print_summary GRAPH=$$GRAPH; \
	done
	@echo ""
	@echo "All results written to: $(RESULTS_FILE)"

# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset summary  (internal target)
# ─────────────────────────────────────────────────────────────────────────────

_print_summary:
	@echo "============================================================" | tee -a $(RESULTS_FILE)
	@echo " BENCHMARK SUMMARY"                                           | tee -a $(RESULTS_FILE)
	@echo " Graph: $(GRAPH)"                                             | tee -a $(RESULTS_FILE)
	@echo "============================================================" | tee -a $(RESULTS_FILE)
	@printf "%-10s %-14s %-12s %-14s\n" "Variant" "Time (s)" "Speedup" "Triangles" \
	        | tee -a $(RESULTS_FILE)
	@echo "--------------------------------------------------------------" | tee -a $(RESULTS_FILE)
	@BT=$$(cat .baseline_time.txt); \
	 for tag in baseline v1 v2 v3 v4 v5; do \
		FILE=".$$tag""_out.txt"; \
		if [ -f "$$FILE" ]; then \
			TIME=$$(grep "^RESULT" $$FILE | awk '{print $$3}'); \
			TRI=$$(grep  "^RESULT" $$FILE | awk '{print $$4}'); \
			if [ "$$tag" = "baseline" ]; then SP="1.00x"; \
			else SP=$$(awk "BEGIN{printf \"%.2fx\", $$BT/$$TIME}"); fi; \
			printf "%-10s %-14s %-12s %-14s\n" "$$tag" "$$TIME" "$$SP" "$$TRI" \
			       | tee -a $(RESULTS_FILE); \
		fi; \
	 done
	@echo "--------------------------------------------------------------" | tee -a $(RESULTS_FILE)
	@echo ""                                                               | tee -a $(RESULTS_FILE)
	@echo "--- Best Performing Variant ---"                               | tee -a $(RESULTS_FILE)
	@BEST="baseline"; BEST_SP=1.0; \
	 BT=$$(cat .baseline_time.txt); \
	 for tag in v1 v2 v3 v4 v5; do \
		FILE=".$$tag""_out.txt"; \
		if [ -f "$$FILE" ]; then \
			TIME=$$(grep "^RESULT" $$FILE | awk '{print $$3}'); \
			SP=$$(awk "BEGIN{printf \"%.4f\", $$BT/$$TIME}"); \
			BETTER=$$(awk "BEGIN{print ($$SP > $$BEST_SP) ? 1 : 0}"); \
			if [ "$$BETTER" = "1" ]; then BEST=$$tag; BEST_SP=$$SP; fi; \
		fi; \
	 done; \
	 echo "  Winner: $$BEST (speedup = $${BEST_SP}x)" | tee -a $(RESULTS_FILE); \
	 echo ""                                           | tee -a $(RESULTS_FILE); \
	 echo "  Analysis:"                                | tee -a $(RESULTS_FILE); \
	 if [ "$$BEST" = "v4" ] || [ "$$BEST" = "v5" ]; then \
		echo "  The Hybrid variant wins because it exploits both levels of"    | tee -a $(RESULTS_FILE); \
		echo "  parallelism: AVX2 SIMD reduces intersection cost per thread,"  | tee -a $(RESULTS_FILE); \
		echo "  while OpenMP distributes vertices across all 8 HW threads."    | tee -a $(RESULTS_FILE); \
	 elif [ "$$BEST" = "v3" ]; then \
		echo "  OpenMP threading wins: graph fits in L3 cache and"             | tee -a $(RESULTS_FILE); \
		echo "  thread-level parallelism is the dominant factor."              | tee -a $(RESULTS_FILE); \
	 elif [ "$$BEST" = "v1" ]; then \
		echo "  AVX2 intrinsics win: large adjacency lists fill the 8-lane"    | tee -a $(RESULTS_FILE); \
		echo "  register, making SIMD the dominant speedup source."            | tee -a $(RESULTS_FILE); \
	 fi
	@echo ""                                                                           | tee -a $(RESULTS_FILE)
	@echo "  SIMD details: AVX2 — 256-bit registers — 8 x int32 lanes."               | tee -a $(RESULTS_FILE)
	@echo "  Memory: i5-10310U L3 = 6 MiB shared across 4 cores."                    | tee -a $(RESULTS_FILE)
	@echo "  Threads: 8 (4 physical x 2 HyperThreading), dynamic chunk=16."           | tee -a $(RESULTS_FILE)
	@echo "============================================================"               | tee -a $(RESULTS_FILE)
	@echo ""                                                                           | tee -a $(RESULTS_FILE)

# ─────────────────────────────────────────────────────────────────────────────
# Clean
# ─────────────────────────────────────────────────────────────────────────────

clean:
	rm -f $(BINARIES) \
	      .baseline_out.txt .v1_out.txt .v2_out.txt .v3_out.txt \
	      .v4_out.txt .v5_out.txt .baseline_time.txt \
	      $(RESULTS_FILE)
	@echo "Cleaned."

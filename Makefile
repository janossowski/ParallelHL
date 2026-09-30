CXX      ?= g++
NVCC     ?= nvcc
CXXFLAGS ?= -O2 -std=c++17 -fopenmp
NVCCFLAGS ?= -O2 -std=c++17 -Xcompiler -fopenmp

BIN := bin

.PHONY: all cpu gpu clean

all: cpu

# CPU-only builds: hl_benchmark.cu and rphast_benchmark.cu compile as plain
# C++ when not run through nvcc. Only the CH build tool and the CPU query
# paths of the benchmarks are available in this mode.
cpu: $(BIN)/ch $(BIN)/hl_benchmark $(BIN)/rphast_benchmark

$(BIN)/ch: ch/ch.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN)/hl_benchmark: hl/hl_benchmark.cu | $(BIN)
	$(CXX) $(CXXFLAGS) -x c++ -o $@ $<

$(BIN)/rphast_benchmark: rphast/rphast_benchmark.cu | $(BIN)
	$(CXX) $(CXXFLAGS) -x c++ -o $@ $<

# GPU builds: requires nvcc and a CUDA toolkit. Produces the same binary
# names with CUDA kernels enabled.
gpu: $(BIN)/hl_benchmark_cuda $(BIN)/rphast_benchmark_cuda

$(BIN)/hl_benchmark_cuda: hl/hl_benchmark.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN)/rphast_benchmark_cuda: rphast/rphast_benchmark.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -rf $(BIN)

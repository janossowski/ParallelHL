# Many-to-many shortest paths on road networks: CH, Hub Labeling, RPHAST

These three programs allow you to build a contraction hierarchy from a
DIMACS graph, then answer many-to-many shortest-path queries against it
using either Hub Labeling, CH or GPU-accelerated RPHAST.

```
ch/ch.cpp                --> reads a DIMACS .gr file, contracts it, writes
                              a CH text file (CH_TEXT_V2 format)
hl/hl_benchmark.cu       --> builds/queries Hub Labels from a CH text file
rphast/rphast_benchmark.cu --> queries a CH text file directly via
                              (GPU) RPHAST or parallel CH
common/                  --> code shared by hl/ and rphast/
```

## Layout

- `ch/ch.cpp` — standalone; builds and validates a contraction hierarchy
  and writes it out as CH text.
- `hl/hl_benchmark.cu` — consumes CH text, builds on-disk Hub Labels, and
  benchmarks CPU and GPU many-to-many queries against them.
- `rphast/rphast_benchmark.cu` — consumes CH text directly and benchmarks
  CPU and GPU RPHAST / parallel-CH many-to-many queries.
- `common/`:
  - `graph_types.hpp` — `Vertex`/`Weight`/`Offset` aliases, `INF`, and the
    CSR graph representation (`Edge`, `CSRGraph`, `CSRBuilder`,
    `reverse_graph()`).
  - `graph_io.hpp` — CH-text line skipping (`next_data_line()`), vertex-ID
    file parsing (`read_vertex_id_file()`), CSV field escaping
    (`csv_escape()`).
  - `query_utils.hpp` — random query generation
    (`generate_random_vertices()`) and result-checksum helpers
    (`mix_query_value()`, `distance_result_fingerprint()`).
  - `cuda_utils.cuh` — `cuda_check()`, a thin CUDA error-checking helper;
    inert outside `nvcc` builds.

## Building

Both `hl_benchmark.cu` and `rphast_benchmark.cu` can be compiled as plain
C++ files: every CUDA-specific construct (`__global__`, `<<<...>>>`,
`cuda_runtime.h`) is behind `#ifdef __CUDACC__`, so a CPU-only build with
a regular C++ compiler works and simply skips the GPU query backends.

```
make cpu   # builds bin/ch, bin/hl_benchmark, bin/rphast_benchmark with g++
make gpu   # builds bin/hl_benchmark_cuda, bin/rphast_benchmark_cuda with nvcc
```

(`CXX`/`NVCC`/`CXXFLAGS`/`NVCCFLAGS` in the `Makefile` can be overridden,
e.g. `make gpu NVCCFLAGS="-O3 -std=c++17 -arch=sm_80 -Xcompiler -fopenmp"`
for a specific GPU architecture.)

## Typical workflow

```
./bin/ch graph.gr --write-ch graph.ch.txt --no-stdin

./bin/hl_benchmark build graph.ch.txt graph.hlbin
./bin/hl_benchmark query graph.hlbin sources.txt targets.txt --backend cpu

./bin/rphast_benchmark query graph.ch.txt sources.txt targets.txt --backend cpu
```
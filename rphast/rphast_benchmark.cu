
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include "../common/cuda_utils.cuh"
#include "../common/graph_io.hpp"
#include "../common/graph_types.hpp"
#include "../common/query_utils.hpp"

static constexpr Vertex INVALID_VERTEX = std::numeric_limits<Vertex>::max();
static constexpr unsigned G32_SOURCE_LANES = 32;


struct CHQueryGraph {
    std::vector<std::uint32_t> rank;
    std::vector<std::uint32_t> level;
    std::vector<Vertex> vertex_by_rank;
    CSRGraph upward;
    CSRGraph downward;
    CSRGraph reverse_downward;
};

static std::uint64_t csr_logical_storage_bytes(const CSRGraph& graph) {
    return static_cast<std::uint64_t>(graph.first.size()) * sizeof(Offset) +
           static_cast<std::uint64_t>(graph.head.size()) * sizeof(Vertex) +
           static_cast<std::uint64_t>(graph.weight.size()) * sizeof(Weight);
}

static std::uint64_t ch_logical_storage_bytes(const CHQueryGraph& ch) {
    return static_cast<std::uint64_t>(ch.rank.size()) * sizeof(std::uint32_t) +
           static_cast<std::uint64_t>(ch.level.size()) * sizeof(std::uint32_t) +
           static_cast<std::uint64_t>(ch.vertex_by_rank.size()) * sizeof(Vertex) +
           csr_logical_storage_bytes(ch.upward) +
           csr_logical_storage_bytes(ch.downward) +
           csr_logical_storage_bytes(ch.reverse_downward);
}

static CHQueryGraph read_ch_text(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open CH file: " + path);
    }

    std::string line;
    if (!next_data_line(input, line) || line != "CH_TEXT_V2") {
        throw std::runtime_error("Expected CH_TEXT_V2");
    }

    if (!next_data_line(input, line)) {
        throw std::runtime_error("Missing CH header");
    }

    std::istringstream header_stream(line);
    std::string p;
    std::string ch;
    std::string n_word;
    std::string augmented_word;
    std::string upward_word;
    std::string downward_word;
    std::uint64_t n64 = 0;
    std::uint64_t augmented = 0;
    std::uint64_t upward_edges = 0;
    std::uint64_t downward_edges = 0;

    header_stream
        >> p >> ch >> n_word >> n64
        >> augmented_word >> augmented
        >> upward_word >> upward_edges
        >> downward_word >> downward_edges;

    if (!header_stream ||
        p != "p" ||
        ch != "ch" ||
        n_word != "n" ||
        augmented_word != "augmented_arcs" ||
        upward_word != "upward_edges" ||
        downward_word != "downward_edges") {

        throw std::runtime_error("Malformed CH p-line");
    }

    if (n64 > std::numeric_limits<Vertex>::max()) {
        throw std::runtime_error("Too many vertices for 32-bit vertex IDs");
    }

    const std::size_t n = static_cast<std::size_t>(n64);
    std::vector<std::uint32_t> rank(
        n,
        std::numeric_limits<std::uint32_t>::max());
    std::vector<std::uint32_t> level(n, 0);

    bool got_upward = false;
    bool got_downward = false;
    CSRGraph upward;
    CSRGraph downward;

    while (next_data_line(input, line)) {
        std::istringstream line_stream(line);
        std::string token;
        line_stream >> token;

        if (token == "begin_vertices") {
            std::uint64_t count = 0;
            line_stream >> count;
            if (count != n64) {
                throw std::runtime_error("Vertex count mismatch");
            }

            std::uint64_t seen = 0;
            while (next_data_line(input, line)) {
                std::istringstream vertex_stream(line);
                std::string kind;
                vertex_stream >> kind;
                if (kind == "end_vertices") {
                    break;
                }
                if (kind != "v") {
                    throw std::runtime_error(
                        "Expected v or end_vertices");
                }

                std::uint64_t id = 0;
                std::uint64_t r = 0;
                std::uint64_t lev = 0;
                vertex_stream >> id >> r >> lev;

                if (!vertex_stream || id == 0 || id > n64 || r >= n64) {
                    throw std::runtime_error("Malformed vertex record");
                }

                rank[id - 1] = static_cast<std::uint32_t>(r);
                level[id - 1] = static_cast<std::uint32_t>(lev);
                ++seen;
            }

            if (seen != n64) {
                throw std::runtime_error("Incomplete vertex section");
            }
            continue;
        }

        if (token == "begin_augmented_arcs") {
            while (next_data_line(input, line)) {
                std::istringstream arc_stream(line);
                std::string kind;
                arc_stream >> kind;
                if (kind == "end_augmented_arcs") {
                    break;
                }
            }
            continue;
        }

        if (token == "begin_csr") {
            std::string direction;
            std::string n_tag;
            std::string edges_tag;
            std::uint64_t section_n = 0;
            std::uint64_t section_edges = 0;

            line_stream
                >> direction >> n_tag >> section_n
                >> edges_tag >> section_edges;

            if (!line_stream ||
                n_tag != "n" ||
                edges_tag != "edges" ||
                section_n != n64) {

                throw std::runtime_error("Malformed begin_csr line");
            }

            if (direction != "upward" && direction != "downward") {
                throw std::runtime_error("Unknown CSR direction");
            }

            CSRBuilder builder(n, static_cast<std::size_t>(section_edges));

            while (next_data_line(input, line)) {
                std::istringstream block_stream(line);
                std::string kind;
                block_stream >> kind;

                if (kind == "end_csr") {
                    std::string end_direction;
                    block_stream >> end_direction;
                    if (end_direction != direction) {
                        throw std::runtime_error(
                            "Mismatched end_csr direction");
                    }
                    break;
                }

                if (kind != "b") {
                    throw std::runtime_error("Expected b or end_csr");
                }

                std::uint64_t source_id = 0;
                std::uint64_t degree = 0;
                block_stream >> source_id >> degree;

                if (!block_stream || source_id == 0 || source_id > n64) {
                    throw std::runtime_error("Malformed b record");
                }

                std::vector<Edge> edges;
                edges.reserve(static_cast<std::size_t>(degree));

                for (std::uint64_t i = 0; i < degree; ++i) {
                    if (!next_data_line(input, line)) {
                        throw std::runtime_error(
                            "Unexpected EOF inside CSR block");
                    }

                    std::istringstream edge_stream(line);
                    std::string edge_kind;
                    std::uint64_t target_id = 0;
                    std::uint64_t weight = 0;
                    edge_stream >> edge_kind >> target_id >> weight;

                    if (!edge_stream ||
                        edge_kind != "e" ||
                        target_id == 0 ||
                        target_id > n64) {

                        throw std::runtime_error("Malformed e record");
                    }

                    edges.push_back(Edge{
                        static_cast<Vertex>(target_id - 1),
                        static_cast<Weight>(weight)});
                }

                builder.add_block(
                    static_cast<Vertex>(source_id - 1),
                    edges);
            }

            CSRGraph graph = builder.finish();
            if (direction == "upward") {
                if (section_edges != upward_edges) {
                    throw std::runtime_error(
                        "Upward edge count disagrees with p-line");
                }
                upward = std::move(graph);
                got_upward = true;
            } else {
                if (section_edges != downward_edges) {
                    throw std::runtime_error(
                        "Downward edge count disagrees with p-line");
                }
                downward = std::move(graph);
                got_downward = true;
            }
            continue;
        }

        if (token == "end_ch") {
            break;
        }
    }

    if (!got_upward || !got_downward) {
        throw std::runtime_error(
            "CH file must contain upward and downward CSR sections");
    }

    std::vector<Vertex> vertex_by_rank(n, INVALID_VERTEX);

    for (std::size_t v = 0; v < n; ++v) {
        if (rank[v] == std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("Missing rank for a vertex");
        }

        const std::uint32_t r = rank[v];
        if (vertex_by_rank[r] != INVALID_VERTEX) {
            throw std::runtime_error("CH ranks are not a permutation");
        }
        vertex_by_rank[r] = static_cast<Vertex>(v);
    }

    for (Vertex u = 0; u < n; ++u) {
        for (Offset i = upward.first[u]; i < upward.first[u + 1]; ++i) {
            if (!(rank[u] < rank[upward.head[i]])) {
                throw std::runtime_error(
                    "Invalid upward edge rank orientation");
            }
        }

        for (Offset i = downward.first[u]; i < downward.first[u + 1]; ++i) {
            if (!(rank[u] > rank[downward.head[i]])) {
                throw std::runtime_error(
                    "Invalid downward edge rank orientation");
            }
        }
    }

    CSRGraph reverse_downward = reverse_graph(downward);

    return CHQueryGraph{
        std::move(rank),
        std::move(level),
        std::move(vertex_by_rank),
        std::move(upward),
        std::move(downward),
        std::move(reverse_downward)};
}

struct RestrictedDownward {
    std::vector<Vertex> vertex;
    std::vector<Offset> first;
    std::vector<Vertex> head;
    std::vector<Weight> weight;
    std::vector<Vertex> target_index;
};

static RestrictedDownward build_restricted_downward(
    const CHQueryGraph& ch,
    const std::vector<Vertex>& targets) {

    const std::size_t n = ch.upward.num_vertices();
    std::vector<std::uint8_t> selected(n, 0);
    std::vector<Vertex> queue;
    const std::size_t reserve_hint =
        targets.size() > (std::numeric_limits<std::size_t>::max() - 16) / 8
            ? n
            : std::min<std::size_t>(n, targets.size() * 8 + 16);
    queue.reserve(reserve_hint);

    for (Vertex target : targets) {
        if (target >= n) {
            throw std::runtime_error("Target vertex is out of range");
        }
        if (!selected[target]) {
            selected[target] = 1;
            queue.push_back(target);
        }
    }

    for (std::size_t q = 0; q < queue.size(); ++q) {
        const Vertex v = queue[q];
        for (Offset i = ch.reverse_downward.first[v];
             i < ch.reverse_downward.first[v + 1];
             ++i) {

            const Vertex u = ch.reverse_downward.head[i];
            if (!selected[u]) {
                selected[u] = 1;
                queue.push_back(u);
            }
        }
    }

    RestrictedDownward restricted;
    restricted.vertex.reserve(queue.size());

    for (std::size_t r = n; r-- > 0;) {
        const Vertex v = ch.vertex_by_rank[r];
        if (selected[v]) {
            restricted.vertex.push_back(v);
        }
    }

    if (restricted.vertex.size() >
        static_cast<std::size_t>(std::numeric_limits<Vertex>::max())) {
        throw std::runtime_error(
            "Restricted graph is too large for 32-bit restricted indices");
    }

    std::vector<Vertex> restricted_index(n, INVALID_VERTEX);
    for (std::size_t i = 0; i < restricted.vertex.size(); ++i) {
        restricted_index[restricted.vertex[i]] = static_cast<Vertex>(i);
    }

    restricted.first.resize(restricted.vertex.size() + 1, 0);

    for (std::size_t local_u = 0;
         local_u < restricted.vertex.size();
         ++local_u) {

        const Vertex u = restricted.vertex[local_u];

        for (Offset e = ch.downward.first[u];
             e < ch.downward.first[u + 1];
             ++e) {

            const Vertex v = ch.downward.head[e];
            const Vertex local_v = restricted_index[v];

            if (local_v == INVALID_VERTEX) {
                continue;
            }

            if (local_v <= local_u) {
                throw std::runtime_error(
                    "Restricted downward arc violates descending-rank order");
            }

            restricted.head.push_back(local_v);
            restricted.weight.push_back(ch.downward.weight[e]);
        }

        restricted.first[local_u + 1] = restricted.head.size();
    }

    restricted.target_index.reserve(targets.size());
    for (Vertex target : targets) {
        const Vertex local = restricted_index[target];
        if (local == INVALID_VERTEX) {
            throw std::runtime_error(
                "Internal error: target missing from restricted graph");
        }
        restricted.target_index.push_back(local);
    }

    return restricted;
}


// GPU-oriented representation for G-RPHAST-32.

struct G32RestrictedLayout {
    std::vector<Vertex> restricted_index;
    std::vector<Offset> incoming_first;
    std::vector<Vertex> incoming_tail;
    std::vector<Weight> incoming_weight;
    std::vector<Offset> level_offsets;
    std::vector<Vertex> level_vertices;
    std::vector<Vertex> target_index;
    std::uint32_t maximum_level = 0;
};

static G32RestrictedLayout build_g32_restricted_layout(
    const RestrictedDownward& restricted,
    std::size_t original_vertex_count) {

    const std::size_t rcount = restricted.vertex.size();
    if (rcount == 0) {
        throw std::runtime_error("G-RPHAST restricted graph is empty");
    }

    G32RestrictedLayout layout;
    layout.restricted_index.assign(original_vertex_count, INVALID_VERTEX);

    for (std::size_t local = 0; local < rcount; ++local) {
        const Vertex original = restricted.vertex[local];
        if (original >= original_vertex_count) {
            throw std::runtime_error("Restricted vertex is out of range");
        }
        layout.restricted_index[original] = static_cast<Vertex>(local);
    }

    // Derive dependency levels from the restricted DAG
    std::vector<std::uint32_t> level(rcount, 0);
    std::uint32_t maximum_level = 0;

    for (std::size_t local_u = rcount; local_u-- > 0;) {
        std::uint32_t best = 0;
        for (Offset e = restricted.first[local_u];
             e < restricted.first[local_u + 1];
             ++e) {

            const Vertex local_v = restricted.head[e];
            if (local_v <= local_u || local_v >= rcount) {
                throw std::runtime_error(
                    "Restricted graph violates descending-rank DAG order");
            }

            if (level[local_v] == std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("G-RPHAST dependency level overflow");
            }

            best = std::max<std::uint32_t>(
                best,
                static_cast<std::uint32_t>(level[local_v] + 1));
        }
        level[local_u] = best;
        maximum_level = std::max(maximum_level, best);
    }

    layout.maximum_level = maximum_level;
    const std::size_t level_count =
        static_cast<std::size_t>(maximum_level) + 1;

    std::vector<Offset> bucket_count(level_count, 0);
    for (std::size_t local = 0; local < rcount; ++local) {
        const std::size_t bucket =
            static_cast<std::size_t>(maximum_level - level[local]);
        ++bucket_count[bucket];
    }

    layout.level_offsets.resize(level_count + 1, 0);
    for (std::size_t bucket = 0; bucket < level_count; ++bucket) {
        layout.level_offsets[bucket + 1] =
            layout.level_offsets[bucket] + bucket_count[bucket];
    }

    layout.level_vertices.resize(rcount, INVALID_VERTEX);
    std::vector<Offset> bucket_cursor = layout.level_offsets;
    for (std::size_t local = 0; local < rcount; ++local) {
        const std::size_t bucket =
            static_cast<std::size_t>(maximum_level - level[local]);
        layout.level_vertices[bucket_cursor[bucket]++] =
            static_cast<Vertex>(local);
    }

    layout.incoming_first.assign(rcount + 1, 0);
    for (std::size_t local_u = 0; local_u < rcount; ++local_u) {
        for (Offset e = restricted.first[local_u];
             e < restricted.first[local_u + 1];
             ++e) {
            const Vertex local_v = restricted.head[e];
            ++layout.incoming_first[static_cast<std::size_t>(local_v) + 1];
        }
    }

    for (std::size_t i = 1; i <= rcount; ++i) {
        layout.incoming_first[i] += layout.incoming_first[i - 1];
    }

    layout.incoming_tail.resize(restricted.head.size());
    layout.incoming_weight.resize(restricted.weight.size());
    std::vector<Offset> incoming_cursor = layout.incoming_first;

    for (std::size_t local_u = 0; local_u < rcount; ++local_u) {
        for (Offset e = restricted.first[local_u];
             e < restricted.first[local_u + 1];
             ++e) {

            const Vertex local_v = restricted.head[e];
            const Offset out = incoming_cursor[local_v]++;
            layout.incoming_tail[out] = static_cast<Vertex>(local_u);
            layout.incoming_weight[out] = restricted.weight[e];

            if (!(level[local_u] > level[local_v])) {
                throw std::runtime_error(
                    "Derived G-RPHAST levels do not respect a downward arc");
            }
        }
    }

    layout.target_index = restricted.target_index;
    return layout;
}

struct G32UpwardSeed {
    Vertex local_vertex = INVALID_VERTEX;
    Weight distance = INF;
};

struct G32SeedRecord {
    Vertex local_vertex = INVALID_VERTEX;
    std::uint32_t lane = 0;
    Weight distance = INF;
};

static_assert(sizeof(G32SeedRecord) == 16,
              "Unexpected G32SeedRecord layout");


struct G32DirectLayout {
    std::size_t restricted_vertex_count = 0;
    std::vector<Offset> incoming_first;
    std::vector<Vertex> incoming_tail;
    std::vector<Weight> incoming_weight;
    std::vector<Offset> level_offsets;
    std::vector<Vertex> level_vertices;
    std::vector<Vertex> target_index;
    std::uint32_t maximum_level = 0;

    std::uint64_t capacity_bytes() const {
        auto bytes = [](std::size_t capacity, std::size_t element_size) {
            if (element_size != 0 &&
                capacity > std::numeric_limits<std::uint64_t>::max() / element_size) {
                return std::numeric_limits<std::uint64_t>::max();
            }
            return static_cast<std::uint64_t>(capacity) * element_size;
        };
        std::uint64_t total = 0;
        auto add = [&total](std::uint64_t value) {
            if (total == std::numeric_limits<std::uint64_t>::max() ||
                value == std::numeric_limits<std::uint64_t>::max() ||
                total > std::numeric_limits<std::uint64_t>::max() - value) {
                total = std::numeric_limits<std::uint64_t>::max();
            } else {
                total += value;
            }
        };
        add(bytes(incoming_first.capacity(), sizeof(Offset)));
        add(bytes(incoming_tail.capacity(), sizeof(Vertex)));
        add(bytes(incoming_weight.capacity(), sizeof(Weight)));
        add(bytes(level_offsets.capacity(), sizeof(Offset)));
        add(bytes(level_vertices.capacity(), sizeof(Vertex)));
        add(bytes(target_index.capacity(), sizeof(Vertex)));
        return total;
    }
};

class G32SelectionWorkspace {
public:
    explicit G32SelectionWorkspace(std::size_t n)
        : selected_epoch_(n, 0),
          local_epoch_(n, 0),
          local_index_(n, INVALID_VERTEX) {}

    void select(const CHQueryGraph& ch, const std::vector<Vertex>& targets) {
        const std::size_t n = ch.upward.num_vertices();
        if (selected_epoch_.size() != n) {
            throw std::runtime_error("G32 selection workspace size mismatch");
        }

        advance_epoch();
        selected_vertices_.clear();

        const std::size_t reserve_hint =
            targets.size() > (std::numeric_limits<std::size_t>::max() - 16) / 8
                ? n
                : std::min<std::size_t>(n, targets.size() * 8 + 16);
        if (selected_vertices_.capacity() < reserve_hint) {
            selected_vertices_.reserve(reserve_hint);
        }

        for (Vertex target : targets) {
            if (target >= n) {
                throw std::runtime_error("Target vertex is out of range");
            }
            if (selected_epoch_[target] != epoch_) {
                selected_epoch_[target] = epoch_;
                selected_vertices_.push_back(target);
            }
        }

        for (std::size_t q = 0; q < selected_vertices_.size(); ++q) {
            const Vertex v = selected_vertices_[q];
            for (Offset e = ch.reverse_downward.first[v];
                 e < ch.reverse_downward.first[v + 1]; ++e) {
                const Vertex u = ch.reverse_downward.head[e];
                if (selected_epoch_[u] != epoch_) {
                    selected_epoch_[u] = epoch_;
                    selected_vertices_.push_back(u);
                }
            }
        }

        if (selected_vertices_.size() >
            static_cast<std::size_t>(std::numeric_limits<Vertex>::max())) {
            throw std::runtime_error(
                "Restricted graph is too large for 32-bit restricted indices");
        }

        used_full_rank_scan_ = selected_vertices_.size() > n / 16;
        if (!used_full_rank_scan_) {
            std::sort(
                selected_vertices_.begin(),
                selected_vertices_.end(),
                [&ch](Vertex a, Vertex b) {
                    return ch.rank[a] > ch.rank[b];
                });
        } else {
            ordered_vertices_.clear();
            if (ordered_vertices_.capacity() < selected_vertices_.size()) {
                ordered_vertices_.reserve(selected_vertices_.size());
            }
            for (std::size_t r = n; r-- > 0;) {
                const Vertex v = ch.vertex_by_rank[r];
                if (selected_epoch_[v] == epoch_) {
                    ordered_vertices_.push_back(v);
                }
            }
            selected_vertices_.swap(ordered_vertices_);
            ordered_vertices_.clear();
        }

        for (std::size_t local = 0; local < selected_vertices_.size(); ++local) {
            const Vertex v = selected_vertices_[local];
            local_epoch_[v] = epoch_;
            local_index_[v] = static_cast<Vertex>(local);
        }
    }

    bool contains(Vertex v) const {
        return v < local_epoch_.size() && local_epoch_[v] == epoch_;
    }

    Vertex local_index(Vertex v) const {
        if (!contains(v)) {
            return INVALID_VERTEX;
        }
        return local_index_[v];
    }

    const std::vector<Vertex>& selected_vertices() const {
        return selected_vertices_;
    }

    bool used_full_rank_scan() const { return used_full_rank_scan_; }

    std::uint64_t capacity_bytes() const {
        return static_cast<std::uint64_t>(selected_epoch_.capacity()) * sizeof(std::uint32_t) +
               static_cast<std::uint64_t>(local_epoch_.capacity()) * sizeof(std::uint32_t) +
               static_cast<std::uint64_t>(local_index_.capacity()) * sizeof(Vertex) +
               static_cast<std::uint64_t>(selected_vertices_.capacity()) * sizeof(Vertex) +
               static_cast<std::uint64_t>(ordered_vertices_.capacity()) * sizeof(Vertex);
    }

private:
    void advance_epoch() {
        if (epoch_ == std::numeric_limits<std::uint32_t>::max()) {
            std::fill(selected_epoch_.begin(), selected_epoch_.end(), 0);
            std::fill(local_epoch_.begin(), local_epoch_.end(), 0);
            epoch_ = 1;
        } else {
            ++epoch_;
            if (epoch_ == 0) {
                epoch_ = 1;
            }
        }
    }

    std::vector<std::uint32_t> selected_epoch_;
    std::vector<std::uint32_t> local_epoch_;
    std::vector<Vertex> local_index_;
    std::vector<Vertex> selected_vertices_;
    std::vector<Vertex> ordered_vertices_;
    std::uint32_t epoch_ = 0;
    bool used_full_rank_scan_ = false;
};

static G32DirectLayout build_g32_direct_layout(
    const CHQueryGraph& ch,
    const std::vector<Vertex>& targets,
    const G32SelectionWorkspace& selection) {

    const auto& vertices = selection.selected_vertices();
    const std::size_t rcount = vertices.size();
    if (rcount == 0) {
        throw std::runtime_error("G-RPHAST restricted graph is empty");
    }

    G32DirectLayout layout;
    layout.restricted_vertex_count = rcount;
    layout.incoming_first.resize(rcount + 1, 0);

    for (std::size_t local_v = 0; local_v < rcount; ++local_v) {
        const Vertex v = vertices[local_v];
        const Offset degree =
            ch.reverse_downward.first[v + 1] - ch.reverse_downward.first[v];
        if (layout.incoming_first[local_v] >
            std::numeric_limits<Offset>::max() - degree) {
            throw std::runtime_error("Restricted incoming edge count overflow");
        }
        layout.incoming_first[local_v + 1] =
            layout.incoming_first[local_v] + degree;
    }

    const Offset edge_count_offset = layout.incoming_first.back();
    if (edge_count_offset >
        static_cast<Offset>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("Restricted incoming graph is too large");
    }
    const std::size_t edge_count = static_cast<std::size_t>(edge_count_offset);
    layout.incoming_tail.resize(edge_count);
    layout.incoming_weight.resize(edge_count);

    std::vector<std::uint32_t> depth(rcount, 0);
    std::uint32_t maximum_depth = 0;

    for (std::size_t local_v = 0; local_v < rcount; ++local_v) {
        const Vertex v = vertices[local_v];
        std::uint32_t best_depth = 0;
        Offset out = layout.incoming_first[local_v];

        for (Offset e = ch.reverse_downward.first[v];
             e < ch.reverse_downward.first[v + 1]; ++e, ++out) {
            const Vertex u = ch.reverse_downward.head[e];
            const Vertex local_u = selection.local_index(u);
            if (local_u == INVALID_VERTEX || local_u >= local_v) {
                throw std::runtime_error(
                    "Direct G32 layout violates predecessor-closed rank order");
            }
            layout.incoming_tail[out] = local_u;
            layout.incoming_weight[out] = ch.reverse_downward.weight[e];

            if (depth[local_u] == std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("G32 dependency depth overflow");
            }
            best_depth = std::max(
                best_depth,
                static_cast<std::uint32_t>(depth[local_u] + 1));
        }

        depth[local_v] = best_depth;
        maximum_depth = std::max(maximum_depth, best_depth);
    }

    layout.maximum_level = maximum_depth;
    const std::size_t level_count =
        static_cast<std::size_t>(maximum_depth) + 1;
    std::vector<Offset> counts(level_count, 0);
    for (std::uint32_t d : depth) {
        ++counts[static_cast<std::size_t>(d)];
    }

    layout.level_offsets.resize(level_count + 1, 0);
    for (std::size_t level = 0; level < level_count; ++level) {
        layout.level_offsets[level + 1] =
            layout.level_offsets[level] + counts[level];
    }

    layout.level_vertices.resize(rcount, INVALID_VERTEX);
    std::vector<Offset> cursor = layout.level_offsets;
    for (std::size_t local = 0; local < rcount; ++local) {
        const std::size_t d = static_cast<std::size_t>(depth[local]);
        layout.level_vertices[cursor[d]++] = static_cast<Vertex>(local);
    }

    layout.target_index.reserve(targets.size());
    for (Vertex target : targets) {
        const Vertex local = selection.local_index(target);
        if (local == INVALID_VERTEX) {
            throw std::runtime_error("Target missing from direct G32 layout");
        }
        layout.target_index.push_back(local);
    }

    return layout;
}

class G32ReusableUpwardWorkspace {
public:
    using QueueEntry = std::pair<Weight, Vertex>;

    explicit G32ReusableUpwardWorkspace(std::size_t n)
        : distance_(n, INF) {}

    void run(
        const CHQueryGraph& ch,
        const G32SelectionWorkspace& selection,
        Vertex source,
        std::vector<G32UpwardSeed>& output) {

        for (Vertex v : touched_) {
            distance_[v] = INF;
        }
        touched_.clear();
        heap_.clear();
        output.clear();

        distance_[source] = 0;
        touched_.push_back(source);
        push_heap_entry({0, source});

        while (!heap_.empty()) {
            const QueueEntry top = pop_heap_entry();
            const Weight du = top.first;
            const Vertex u = top.second;
            if (du != distance_[u]) {
                continue;
            }

            const Vertex local = selection.local_index(u);
            if (local != INVALID_VERTEX) {
                output.push_back(G32UpwardSeed{local, du});
            }

            for (Offset e = ch.upward.first[u];
                 e < ch.upward.first[u + 1]; ++e) {
                const Vertex v = ch.upward.head[e];
                const Weight w = ch.upward.weight[e];
                if (w >= INF || du > INF - w) {
                    continue;
                }
                const Weight candidate = du + w;
                if (candidate < distance_[v]) {
                    if (distance_[v] == INF) {
                        touched_.push_back(v);
                    }
                    distance_[v] = candidate;
                    push_heap_entry({candidate, v});
                }
            }
        }
    }

    std::uint64_t capacity_bytes() const {
        return static_cast<std::uint64_t>(distance_.capacity()) * sizeof(Weight) +
               static_cast<std::uint64_t>(touched_.capacity()) * sizeof(Vertex) +
               static_cast<std::uint64_t>(heap_.capacity()) * sizeof(QueueEntry);
    }

private:

    void push_heap_entry(QueueEntry entry) {
        heap_.push_back(entry);
        std::push_heap(heap_.begin(), heap_.end(), std::greater<QueueEntry>());
    }

    QueueEntry pop_heap_entry() {
        std::pop_heap(heap_.begin(), heap_.end(), std::greater<QueueEntry>());
        QueueEntry entry = heap_.back();
        heap_.pop_back();
        return entry;
    }

    std::vector<Weight> distance_;
    std::vector<Vertex> touched_;
    std::vector<QueueEntry> heap_;
};

class G32OptimizedHostWorkspace {
public:
    G32OptimizedHostWorkspace(std::size_t n, int requested_threads)
        : selection(n) {
        if (requested_threads <= 0) {
            throw std::runtime_error("Thread count must be positive");
        }
        upward_workers.reserve(static_cast<std::size_t>(requested_threads));
        for (int i = 0; i < requested_threads; ++i) {
            upward_workers.push_back(
                std::make_unique<G32ReusableUpwardWorkspace>(n));
        }
    }

    std::uint64_t capacity_bytes() const {
        std::uint64_t total = selection.capacity_bytes();
        for (const auto& worker : upward_workers) {
            total += worker->capacity_bytes();
        }
        for (const auto& lane : lane_seeds) {
            total += static_cast<std::uint64_t>(lane.capacity()) * sizeof(G32UpwardSeed);
        }
        total += static_cast<std::uint64_t>(flat_seeds.capacity()) * sizeof(G32SeedRecord);
        return total;
    }

    G32SelectionWorkspace selection;
    std::vector<std::unique_ptr<G32ReusableUpwardWorkspace>> upward_workers;
    std::array<std::vector<G32UpwardSeed>, G32_SOURCE_LANES> lane_seeds;
    std::vector<G32SeedRecord> flat_seeds;
};

static inline bool safe_relax_value(
    Weight base,
    Weight edge_weight,
    Weight& destination) {

    if (base == INF || edge_weight >= INF || base > INF - edge_weight) {
        return false;
    }

    const Weight candidate = base + edge_weight;
    if (candidate < destination) {
        destination = candidate;
        return true;
    }
    return false;
}


static void print_distance_result(
    Vertex source,
    Vertex target,
    Weight distance) {

    std::cout
        << (static_cast<std::uint64_t>(source) + 1)
        << ' '
        << (static_cast<std::uint64_t>(target) + 1)
        << ' ';

    if (distance == INF) {
        std::cout << "unreachable 0\n";
    } else {
        std::cout << distance << " 0\n";
    }
}

class RPHASTDijkstraWorkspace {
public:
    explicit RPHASTDijkstraWorkspace(std::size_t n)
        : distance_(n, INF) {}

    void run(
        const CHQueryGraph& ch,
        const RestrictedDownward& restricted,
        Vertex source,
        Weight* target_distances) {

        for (Vertex v : touched_) {
            distance_[v] = INF;
        }
        touched_.clear();

        using QueueEntry = std::pair<Weight, Vertex>;
        std::priority_queue<
            QueueEntry,
            std::vector<QueueEntry>,
            std::greater<QueueEntry>> queue;

        distance_[source] = 0;
        touched_.push_back(source);
        queue.emplace(0, source);

        while (!queue.empty()) {
            const auto [du, u] = queue.top();
            queue.pop();

            if (du != distance_[u]) {
                continue;
            }

            for (Offset e = ch.upward.first[u];
                 e < ch.upward.first[u + 1];
                 ++e) {

                const Vertex v = ch.upward.head[e];
                const Weight w = ch.upward.weight[e];

                if (du == INF || w >= INF || du > INF - w) {
                    continue;
                }

                const Weight candidate = du + w;
                if (candidate < distance_[v]) {
                    if (distance_[v] == INF) {
                        touched_.push_back(v);
                    }
                    distance_[v] = candidate;
                    queue.emplace(candidate, v);
                }
            }
        }

        for (std::size_t local_u = 0;
             local_u < restricted.vertex.size();
             ++local_u) {

            const Vertex u = restricted.vertex[local_u];
            const Weight du = distance_[u];

            if (du == INF) {
                continue;
            }

            for (Offset e = restricted.first[local_u];
                 e < restricted.first[local_u + 1];
                 ++e) {

                const Vertex local_v = restricted.head[e];
                const Vertex v = restricted.vertex[local_v];
                const Weight w = restricted.weight[e];

                if (du == INF || w >= INF || du > INF - w) {
                    continue;
                }

                const Weight candidate = du + w;
                if (candidate < distance_[v]) {
                    if (distance_[v] == INF) {
                        touched_.push_back(v);
                    }
                    distance_[v] = candidate;
                }
            }
        }

        for (std::size_t i = 0; i < restricted.target_index.size(); ++i) {
            const Vertex target =
                restricted.vertex[restricted.target_index[i]];
            target_distances[i] = distance_[target];
        }
    }

private:
    std::vector<Weight> distance_;
    std::vector<Vertex> touched_;
};



// CPU side of G-RPHAST-32: run the ordinary CH upward search, then retain only
// finalized distances whose vertices occur in the target-restricted graph.

class G32UpwardSeedWorkspace {
public:
    explicit G32UpwardSeedWorkspace(std::size_t n)
        : distance_(n, INF) {}

    void run(
        const CHQueryGraph& ch,
        const std::vector<Vertex>& restricted_index,
        Vertex source,
        std::vector<G32UpwardSeed>& output) {

        if (restricted_index.size() != distance_.size()) {
            throw std::runtime_error(
                "G32 restricted-index map has the wrong vertex count");
        }

        for (Vertex v : touched_) {
            distance_[v] = INF;
        }
        touched_.clear();
        output.clear();

        using QueueEntry = std::pair<Weight, Vertex>;
        std::priority_queue<
            QueueEntry,
            std::vector<QueueEntry>,
            std::greater<QueueEntry>> queue;

        distance_[source] = 0;
        touched_.push_back(source);
        queue.emplace(0, source);

        while (!queue.empty()) {
            const auto [du, u] = queue.top();
            queue.pop();
            if (du != distance_[u]) {
                continue;
            }

            for (Offset e = ch.upward.first[u];
                 e < ch.upward.first[u + 1];
                 ++e) {

                const Vertex v = ch.upward.head[e];
                const Weight w = ch.upward.weight[e];
                if (w >= INF || du > INF - w) {
                    continue;
                }

                const Weight candidate = du + w;
                if (candidate < distance_[v]) {
                    if (distance_[v] == INF) {
                        touched_.push_back(v);
                    }
                    distance_[v] = candidate;
                    queue.emplace(candidate, v);
                }
            }
        }

        output.reserve(std::max(output.capacity(), touched_.size()));
        for (Vertex v : touched_) {
            const Vertex local = restricted_index[v];
            if (local != INVALID_VERTEX) {
                output.push_back(G32UpwardSeed{local, distance_[v]});
            }
        }
    }

private:
    std::vector<Weight> distance_;
    std::vector<Vertex> touched_;
};

class RPHASTRankSweepWorkspace {
public:
    explicit RPHASTRankSweepWorkspace(std::size_t n)
        : distance_(n, INF) {}

    void run(
        const CHQueryGraph& ch,
        const RestrictedDownward& restricted,
        Vertex source,
        Weight* target_distances) {

        for (Vertex v : touched_) {
            distance_[v] = INF;
        }
        touched_.clear();

        distance_[source] = 0;
        touched_.push_back(source);

        const std::size_t source_rank = ch.rank[source];

        for (std::size_t r = source_rank;
             r < ch.vertex_by_rank.size();
             ++r) {

            const Vertex u = ch.vertex_by_rank[r];
            const Weight du = distance_[u];
            if (du == INF) {
                continue;
            }

            for (Offset e = ch.upward.first[u];
                 e < ch.upward.first[u + 1];
                 ++e) {

                const Vertex v = ch.upward.head[e];
                const Weight old_distance = distance_[v];
                if (safe_relax_value(
                        du,
                        ch.upward.weight[e],
                        distance_[v]) &&
                    old_distance == INF) {
                    touched_.push_back(v);
                }
            }
        }

        for (std::size_t local_u = 0;
             local_u < restricted.vertex.size();
             ++local_u) {

            const Vertex u = restricted.vertex[local_u];
            const Weight du = distance_[u];
            if (du == INF) {
                continue;
            }

            for (Offset e = restricted.first[local_u];
                 e < restricted.first[local_u + 1];
                 ++e) {

                const Vertex local_v = restricted.head[e];
                const Vertex v = restricted.vertex[local_v];
                const Weight old_distance = distance_[v];

                if (safe_relax_value(
                        du,
                        restricted.weight[e],
                        distance_[v]) &&
                    old_distance == INF) {
                    touched_.push_back(v);
                }
            }
        }

        for (std::size_t i = 0; i < restricted.target_index.size(); ++i) {
            const Vertex target =
                restricted.vertex[restricted.target_index[i]];
            target_distances[i] = distance_[target];
        }
    }

private:
    std::vector<Weight> distance_;
    std::vector<Vertex> touched_;
};

enum class CPUQueryMode {
    Dijkstra,
    RankSweep
};

static const char* cpu_query_mode_name(CPUQueryMode mode) {
    return mode == CPUQueryMode::Dijkstra ? "dijkstra" : "rank-sweep";
}

static void run_many_to_many_cpu(
    const CHQueryGraph& ch,
    const RestrictedDownward& restricted,
    const std::vector<Vertex>& sources,
    const std::vector<Vertex>& targets,
    bool quiet,
    int requested_threads,
    CPUQueryMode mode,
    double selection_seconds) {

    if (sources.empty() || targets.empty()) {
        throw std::runtime_error(
            "Source and target sets must both be non-empty");
    }

    if (requested_threads <= 0) {
        throw std::runtime_error("Thread count must be positive");
    }

#ifndef _OPENMP
    if (requested_threads != 1) {
        throw std::runtime_error(
            "This executable was built without OpenMP support; use --threads 1");
    }
#endif

    if (sources.size() >
        std::numeric_limits<std::uint64_t>::max() / targets.size()) {
        throw std::runtime_error(
            "Number of source-target pairs overflows uint64_t");
    }

    const std::uint64_t pair_count =
        static_cast<std::uint64_t>(sources.size()) *
        static_cast<std::uint64_t>(targets.size());

    std::vector<Weight> results;
    if (!quiet) {
        if (pair_count >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error(
                "Too many results to materialize; use --quiet");
        }
        results.resize(static_cast<std::size_t>(pair_count), INF);
    }

    int actual_threads = 1;
#ifdef _OPENMP
    omp_set_dynamic(0);
#pragma omp parallel num_threads(requested_threads)
    {
#pragma omp single
        actual_threads = omp_get_num_threads();
    }
#endif

    std::vector<std::uint64_t> worker_checksums(
        static_cast<std::size_t>(actual_threads), 0);
    std::vector<std::uint64_t> worker_unreachable(
        static_cast<std::size_t>(actual_threads), 0);

    std::cerr
        << "RPHAST CPU many-to-many: "
        << sources.size() << " sources x "
        << targets.size() << " targets = "
        << pair_count << " pairs\n"
        << "upward phase: " << cpu_query_mode_name(mode) << "\n"
        << "query workers: " << actual_threads
#ifdef _OPENMP
        << " (OpenMP)\n";
#else
        << " (serial)\n";
#endif

    const auto start = std::chrono::steady_clock::now();

#ifdef _OPENMP
#pragma omp parallel num_threads(requested_threads)
    {
        const int thread_id = omp_get_thread_num();
        std::uint64_t local_checksum = 0;
        std::uint64_t local_unreachable = 0;
        std::vector<Weight> target_distances(targets.size(), INF);

        std::unique_ptr<RPHASTDijkstraWorkspace> dijkstra_workspace;
        std::unique_ptr<RPHASTRankSweepWorkspace> rank_workspace;
        if (mode == CPUQueryMode::Dijkstra) {
            dijkstra_workspace = std::make_unique<RPHASTDijkstraWorkspace>(
                ch.upward.num_vertices());
        } else {
            rank_workspace = std::make_unique<RPHASTRankSweepWorkspace>(
                ch.upward.num_vertices());
        }

#pragma omp for schedule(static)
        for (std::int64_t source_index = 0;
             source_index < static_cast<std::int64_t>(sources.size());
             ++source_index) {

            const Vertex source =
                sources[static_cast<std::size_t>(source_index)];

            if (mode == CPUQueryMode::Dijkstra) {
                dijkstra_workspace->run(
                    ch,
                    restricted,
                    source,
                    target_distances.data());
            } else {
                rank_workspace->run(
                    ch,
                    restricted,
                    source,
                    target_distances.data());
            }

            const std::size_t result_row =
                static_cast<std::size_t>(source_index) * targets.size();

            for (std::size_t target_index = 0;
                 target_index < targets.size();
                 ++target_index) {

                const Weight distance = target_distances[target_index];
                local_checksum ^=
                    distance_result_fingerprint(
                        source,
                        targets[target_index],
                        distance);

                if (distance == INF) {
                    ++local_unreachable;
                }

                if (!quiet) {
                    results[result_row + target_index] = distance;
                }
            }
        }

        worker_checksums[static_cast<std::size_t>(thread_id)] =
            local_checksum;
        worker_unreachable[static_cast<std::size_t>(thread_id)] =
            local_unreachable;
    }
#else
    {
        std::uint64_t local_checksum = 0;
        std::uint64_t local_unreachable = 0;
        std::vector<Weight> target_distances(targets.size(), INF);
        std::unique_ptr<RPHASTDijkstraWorkspace> dijkstra_workspace;
        std::unique_ptr<RPHASTRankSweepWorkspace> rank_workspace;
        if (mode == CPUQueryMode::Dijkstra) {
            dijkstra_workspace = std::make_unique<RPHASTDijkstraWorkspace>(
                ch.upward.num_vertices());
        } else {
            rank_workspace = std::make_unique<RPHASTRankSweepWorkspace>(
                ch.upward.num_vertices());
        }

        for (std::size_t source_index = 0;
             source_index < sources.size();
             ++source_index) {

            const Vertex source = sources[source_index];
            if (mode == CPUQueryMode::Dijkstra) {
                dijkstra_workspace->run(
                    ch,
                    restricted,
                    source,
                    target_distances.data());
            } else {
                rank_workspace->run(
                    ch,
                    restricted,
                    source,
                    target_distances.data());
            }

            const std::size_t result_row = source_index * targets.size();
            for (std::size_t target_index = 0;
                 target_index < targets.size();
                 ++target_index) {

                const Weight distance = target_distances[target_index];
                local_checksum ^=
                    distance_result_fingerprint(
                        source,
                        targets[target_index],
                        distance);
                if (distance == INF) {
                    ++local_unreachable;
                }
                if (!quiet) {
                    results[result_row + target_index] = distance;
                }
            }
        }

        worker_checksums[0] = local_checksum;
        worker_unreachable[0] = local_unreachable;
    }
#endif

    const auto end = std::chrono::steady_clock::now();
    const double query_seconds =
        std::chrono::duration<double>(end - start).count();

    std::uint64_t checksum = 0;
    std::uint64_t unreachable = 0;
    for (std::size_t i = 0; i < worker_checksums.size(); ++i) {
        checksum ^= worker_checksums[i];
        unreachable += worker_unreachable[i];
    }

    const double total_seconds = selection_seconds + query_seconds;
    const double average_ns =
        query_seconds * 1.0e9 / static_cast<double>(pair_count);

    std::cerr
        << "target selection time: "
        << std::fixed << std::setprecision(6)
        << selection_seconds << " s\n"
        << "query time: " << query_seconds << " s\n"
        << "selection + query time (CH I/O excluded): "
        << total_seconds << " s\n"
        << "average query time: "
        << std::setprecision(2)
        << average_ns << " ns/pair\n"
        << "pairs completed: " << pair_count
        << ", unreachable: " << unreachable
        << ", distance checksum: " << checksum << "\n";

    if (!quiet) {
        for (std::size_t source_index = 0;
             source_index < sources.size();
             ++source_index) {
            for (std::size_t target_index = 0;
                 target_index < targets.size();
                 ++target_index) {

                print_distance_result(
                    sources[source_index],
                    targets[target_index],
                    results[source_index * targets.size() + target_index]);
            }
        }
    }
}


/*
 * Simple regular Contraction Hierarchies baseline.
 *
 * Every s-t pair is an independent bidirectional CH query:
 *   forward:  s over natural upward arcs
 *   backward: t over reversed natural downward arcs
 *
 * Pairs are distributed across OpenMP workers.
 */
struct CHPairQueryCounters {
    std::uint64_t forward_settled = 0;
    std::uint64_t backward_settled = 0;
    std::uint64_t relaxed_arcs = 0;
};

struct CHHeapEntry {
    Weight distance = INF;
    Vertex vertex = INVALID_VERTEX;
};

struct CHHeapGreater {
    bool operator()(const CHHeapEntry& a, const CHHeapEntry& b) const {
        if (a.distance != b.distance) return a.distance > b.distance;
        return a.vertex > b.vertex;
    }
};

class ReusableCHMinHeap {
public:
    void clear() { data_.clear(); }

    void push(Weight distance, Vertex vertex) {
        data_.push_back(CHHeapEntry{distance, vertex});
        std::push_heap(data_.begin(), data_.end(), CHHeapGreater{});
    }

    Weight valid_min(const std::vector<Weight>& distance) {
        discard_stale(distance);
        return data_.empty() ? INF : data_.front().distance;
    }

    bool pop_valid(const std::vector<Weight>& distance, CHHeapEntry& out) {
        discard_stale(distance);
        if (data_.empty()) return false;
        std::pop_heap(data_.begin(), data_.end(), CHHeapGreater{});
        out = data_.back();
        data_.pop_back();
        return true;
    }

    std::uint64_t capacity_bytes() const {
        return static_cast<std::uint64_t>(data_.capacity()) * sizeof(CHHeapEntry);
    }

private:
    void discard_stale(const std::vector<Weight>& distance) {
        while (!data_.empty()) {
            const CHHeapEntry& top = data_.front();
            if (top.vertex < distance.size() &&
                top.distance == distance[top.vertex]) {
                break;
            }
            std::pop_heap(data_.begin(), data_.end(), CHHeapGreater{});
            data_.pop_back();
        }
    }

    std::vector<CHHeapEntry> data_;
};

class CHBidirectionalQueryWorkspace {
public:
    explicit CHBidirectionalQueryWorkspace(std::size_t n)
        : forward_distance_(n, INF), backward_distance_(n, INF) {}

    Weight query(
        const CHQueryGraph& ch,
        Vertex source,
        Vertex target,
        CHPairQueryCounters* counters = nullptr) {

        reset_previous_query();

        if (source >= forward_distance_.size() ||
            target >= backward_distance_.size()) {
            throw std::runtime_error("CH query vertex is out of range");
        }

        forward_distance_[source] = 0;
        backward_distance_[target] = 0;
        forward_touched_.push_back(source);
        backward_touched_.push_back(target);
        forward_heap_.push(0, source);
        backward_heap_.push(0, target);

        Weight best = source == target ? 0 : INF;

        for (;;) {
            const Weight forward_min =
                forward_heap_.valid_min(forward_distance_);
            const Weight backward_min =
                backward_heap_.valid_min(backward_distance_);

            if (forward_min >= best && backward_min >= best) {
                break;
            }
            if (forward_min == INF && backward_min == INF) {
                break;
            }

            if (forward_min <= backward_min) {
                settle_one_side(
                    ch.upward,
                    forward_distance_,
                    backward_distance_,
                    forward_touched_,
                    forward_heap_,
                    best,
                    counters == nullptr ? nullptr : &counters->forward_settled,
                    counters == nullptr ? nullptr : &counters->relaxed_arcs);
            } else {
                settle_one_side(
                    ch.reverse_downward,
                    backward_distance_,
                    forward_distance_,
                    backward_touched_,
                    backward_heap_,
                    best,
                    counters == nullptr ? nullptr : &counters->backward_settled,
                    counters == nullptr ? nullptr : &counters->relaxed_arcs);
            }
        }

        return best;
    }

    std::uint64_t capacity_bytes() const {
        return
            static_cast<std::uint64_t>(forward_distance_.capacity()) * sizeof(Weight) +
            static_cast<std::uint64_t>(backward_distance_.capacity()) * sizeof(Weight) +
            static_cast<std::uint64_t>(forward_touched_.capacity()) * sizeof(Vertex) +
            static_cast<std::uint64_t>(backward_touched_.capacity()) * sizeof(Vertex) +
            forward_heap_.capacity_bytes() +
            backward_heap_.capacity_bytes();
    }

private:
    static bool sum_is_better(Weight a, Weight b, Weight best, Weight& result) {
        if (a == INF || b == INF || a > INF - b) return false;
        const Weight candidate = a + b;
        if (candidate < best) {
            result = candidate;
            return true;
        }
        return false;
    }

    static void settle_one_side(
        const CSRGraph& graph,
        std::vector<Weight>& own_distance,
        const std::vector<Weight>& opposite_distance,
        std::vector<Vertex>& touched,
        ReusableCHMinHeap& heap,
        Weight& best,
        std::uint64_t* settled_counter,
        std::uint64_t* relaxed_arc_counter) {

        CHHeapEntry current;
        if (!heap.pop_valid(own_distance, current)) return;

        const Vertex u = current.vertex;
        const Weight du = current.distance;
        if (settled_counter != nullptr) ++(*settled_counter);

        Weight candidate_best = best;
        if (sum_is_better(du, opposite_distance[u], best, candidate_best)) {
            best = candidate_best;
        }

        for (Offset e = graph.first[u]; e < graph.first[u + 1]; ++e) {
            if (relaxed_arc_counter != nullptr) ++(*relaxed_arc_counter);
            const Vertex v = graph.head[e];
            const Weight w = graph.weight[e];
            if (w >= INF || du > INF - w) continue;
            const Weight candidate = du + w;
            if (candidate < own_distance[v]) {
                if (own_distance[v] == INF) touched.push_back(v);
                own_distance[v] = candidate;
                heap.push(candidate, v);

                Weight joined = best;
                if (sum_is_better(candidate, opposite_distance[v], best, joined)) {
                    best = joined;
                }
            }
        }
    }

    void reset_previous_query() {
        for (Vertex v : forward_touched_) forward_distance_[v] = INF;
        for (Vertex v : backward_touched_) backward_distance_[v] = INF;
        forward_touched_.clear();
        backward_touched_.clear();
        forward_heap_.clear();
        backward_heap_.clear();
    }

    std::vector<Weight> forward_distance_;
    std::vector<Weight> backward_distance_;
    std::vector<Vertex> forward_touched_;
    std::vector<Vertex> backward_touched_;
    ReusableCHMinHeap forward_heap_;
    ReusableCHMinHeap backward_heap_;
};

class ParallelCHWorkspace {
public:
    ParallelCHWorkspace(std::size_t n, int worker_count) {
        if (worker_count <= 0) {
            throw std::runtime_error("CH worker count must be positive");
        }
        workers_.reserve(static_cast<std::size_t>(worker_count));
        for (int i = 0; i < worker_count; ++i) {
            workers_.emplace_back(n);
        }
    }

    CHBidirectionalQueryWorkspace& worker(std::size_t index) {
        return workers_.at(index);
    }

    std::size_t worker_count() const { return workers_.size(); }

    std::uint64_t capacity_bytes() const {
        std::uint64_t total =
            static_cast<std::uint64_t>(workers_.capacity()) *
            sizeof(CHBidirectionalQueryWorkspace);
        for (const auto& worker : workers_) total += worker.capacity_bytes();
        return total;
    }

private:
    std::vector<CHBidirectionalQueryWorkspace> workers_;
};

struct ParallelCHQueryStats {
    std::uint64_t source_count = 0;
    std::uint64_t target_count = 0;
    std::uint64_t pair_count = 0;
    double end_to_end_seconds = 0.0;
    std::uint64_t checksum = 0;
    std::uint64_t unreachable = 0;
    std::uint64_t forward_settled = 0;
    std::uint64_t backward_settled = 0;
    std::uint64_t relaxed_arcs = 0;
    std::uint64_t workspace_bytes_before = 0;
    std::uint64_t workspace_bytes_after = 0;
};

static ParallelCHQueryStats run_many_to_many_parallel_ch(
    const CHQueryGraph& ch,
    ParallelCHWorkspace& workspace,
    const std::vector<Vertex>& sources,
    const std::vector<Vertex>& targets,
    bool quiet,
    int requested_threads,
    bool print_report = true) {

    if (sources.empty() || targets.empty()) {
        throw std::runtime_error("Source and target sets must both be non-empty");
    }
    if (requested_threads <= 0) {
        throw std::runtime_error("Thread count must be positive");
    }
#ifndef _OPENMP
    if (requested_threads != 1) {
        throw std::runtime_error(
            "Parallel CH requires OpenMP for --threads > 1");
    }
#endif
    if (sources.size() >
        std::numeric_limits<std::uint64_t>::max() / targets.size()) {
        throw std::runtime_error("Number of source-target pairs overflows uint64_t");
    }

    const std::uint64_t pair_count =
        static_cast<std::uint64_t>(sources.size()) *
        static_cast<std::uint64_t>(targets.size());
    const std::uint64_t target_count =
        static_cast<std::uint64_t>(targets.size());

    ParallelCHQueryStats stats;
    stats.source_count = static_cast<std::uint64_t>(sources.size());
    stats.target_count = static_cast<std::uint64_t>(targets.size());
    stats.pair_count = pair_count;
    stats.workspace_bytes_before = workspace.capacity_bytes();

    std::vector<Weight> results;
    if (!quiet) {
        if (pair_count > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("Too many CH results to materialize; use --quiet");
        }
        results.resize(static_cast<std::size_t>(pair_count), INF);
    }

    int actual_threads = 1;
#ifdef _OPENMP
    omp_set_dynamic(0);
#pragma omp parallel num_threads(requested_threads)
    {
#pragma omp single
        actual_threads = omp_get_num_threads();
    }
#endif
    if (static_cast<std::size_t>(actual_threads) > workspace.worker_count()) {
        throw std::runtime_error("Persistent CH workspace has too few workers");
    }

    std::vector<std::uint64_t> worker_checksums(
        static_cast<std::size_t>(actual_threads), 0);
    std::vector<std::uint64_t> worker_unreachable(
        static_cast<std::size_t>(actual_threads), 0);
    std::vector<CHPairQueryCounters> worker_counters(
        static_cast<std::size_t>(actual_threads));

    const auto begin = std::chrono::steady_clock::now();

#ifdef _OPENMP
#pragma omp parallel num_threads(requested_threads)
    {
        const int tid = omp_get_thread_num();
        const int tcount = omp_get_num_threads();
        const std::uint64_t tid64 = static_cast<std::uint64_t>(tid);
        const std::uint64_t tc64 = static_cast<std::uint64_t>(tcount);
        const std::uint64_t base = pair_count / tc64;
        const std::uint64_t rem = pair_count % tc64;
        const std::uint64_t pair_begin =
            tid64 * base + std::min<std::uint64_t>(tid64, rem);
        const std::uint64_t pair_end =
            pair_begin + base + (tid64 < rem ? 1 : 0);

        auto& query_workspace = workspace.worker(static_cast<std::size_t>(tid));
        std::uint64_t local_checksum = 0;
        std::uint64_t local_unreachable = 0;
        CHPairQueryCounters local_counters;

        for (std::uint64_t pair = pair_begin; pair < pair_end; ++pair) {
            const std::uint64_t source_index = pair / target_count;
            const std::uint64_t target_index = pair % target_count;
            const Vertex source = sources[static_cast<std::size_t>(source_index)];
            const Vertex target = targets[static_cast<std::size_t>(target_index)];
            const Weight distance = query_workspace.query(
                ch, source, target, &local_counters);

            local_checksum ^=
                distance_result_fingerprint(source, target, distance);
            if (distance == INF) ++local_unreachable;
            if (!quiet) results[static_cast<std::size_t>(pair)] = distance;
        }

        worker_checksums[static_cast<std::size_t>(tid)] = local_checksum;
        worker_unreachable[static_cast<std::size_t>(tid)] = local_unreachable;
        worker_counters[static_cast<std::size_t>(tid)] = local_counters;
    }
#else
    {
        auto& query_workspace = workspace.worker(0);
        CHPairQueryCounters local_counters;
        for (std::uint64_t pair = 0; pair < pair_count; ++pair) {
            const std::uint64_t source_index = pair / target_count;
            const std::uint64_t target_index = pair % target_count;
            const Vertex source = sources[static_cast<std::size_t>(source_index)];
            const Vertex target = targets[static_cast<std::size_t>(target_index)];
            const Weight distance = query_workspace.query(
                ch, source, target, &local_counters);
            worker_checksums[0] ^=
                distance_result_fingerprint(source, target, distance);
            if (distance == INF) ++worker_unreachable[0];
            if (!quiet) results[static_cast<std::size_t>(pair)] = distance;
        }
        worker_counters[0] = local_counters;
    }
#endif

    const auto end = std::chrono::steady_clock::now();
    stats.end_to_end_seconds =
        std::chrono::duration<double>(end - begin).count();

    for (std::size_t i = 0; i < worker_checksums.size(); ++i) {
        stats.checksum ^= worker_checksums[i];
        stats.unreachable += worker_unreachable[i];
        stats.forward_settled += worker_counters[i].forward_settled;
        stats.backward_settled += worker_counters[i].backward_settled;
        stats.relaxed_arcs += worker_counters[i].relaxed_arcs;
    }
    stats.workspace_bytes_after = workspace.capacity_bytes();

    if (print_report) {
        const double ns_per_pair =
            stats.end_to_end_seconds * 1.0e9 / static_cast<double>(pair_count);
        std::cerr
            << std::fixed << std::setprecision(6)
            << "Parallel regular CH query\n"
            << "  workers: " << actual_threads << "\n"
            << "  pairs: " << pair_count << "\n"
            << "  pair-query wall time: " << stats.end_to_end_seconds << " s\n"
            << std::setprecision(2)
            << "  average: " << ns_per_pair << " ns/pair\n"
            << std::setprecision(6)
            << "  forward settled: " << stats.forward_settled << "\n"
            << "  backward settled: " << stats.backward_settled << "\n"
            << "  relaxed arcs: " << stats.relaxed_arcs << "\n"
            << "  persistent workspace after query: "
            << (static_cast<double>(stats.workspace_bytes_after) /
                (1024.0 * 1024.0)) << " MiB\n"
            << "  unreachable: " << stats.unreachable
            << ", checksum: " << stats.checksum << "\n";
    }

    if (!quiet) {
        for (std::size_t si = 0; si < sources.size(); ++si) {
            for (std::size_t ti = 0; ti < targets.size(); ++ti) {
                print_distance_result(
                    sources[si], targets[ti],
                    results[si * targets.size() + ti]);
            }
        }
    }

    return stats;
}

#ifdef __CUDACC__

template <typename T>
class CudaBuffer {
public:
    CudaBuffer() = default;

    explicit CudaBuffer(std::size_t count) {
        allocate(count);
    }

    ~CudaBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    void allocate(std::size_t count) {
        if (data_ != nullptr) {
            cuda_check(cudaFree(data_), "cudaFree");
            data_ = nullptr;
            count_ = 0;
        }
        if (count == 0) {
            return;
        }
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::runtime_error("CUDA allocation size overflows size_t");
        }
        cuda_check(
            cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
            "cudaMalloc");
        count_ = count;
    }

    void upload(const std::vector<T>& host) {
        if (host.size() > count_) {
            throw std::runtime_error("CUDA upload exceeds allocated buffer");
        }
        if (host.empty()) {
            return;
        }
        cuda_check(
            cudaMemcpy(
                data_,
                host.data(),
                host.size() * sizeof(T),
                cudaMemcpyHostToDevice),
            "cudaMemcpy H2D");
    }

    void upload_slice(const T* host, std::size_t count) {
        if (count > count_) {
            throw std::runtime_error("CUDA upload exceeds allocated buffer");
        }
        if (count == 0) {
            return;
        }
        cuda_check(
            cudaMemcpy(
                data_,
                host,
                count * sizeof(T),
                cudaMemcpyHostToDevice),
            "cudaMemcpy H2D slice");
    }

    void download_slice(T* host, std::size_t count) const {
        if (count > count_) {
            throw std::runtime_error("CUDA download exceeds allocated buffer");
        }
        if (count == 0) {
            return;
        }
        cuda_check(
            cudaMemcpy(
                host,
                data_,
                count * sizeof(T),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H slice");
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t size() const { return count_; }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

__device__ static inline std::uint64_t device_mix_query_value(
    std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__device__ static inline std::uint64_t device_distance_fingerprint(
    Vertex source,
    Vertex target,
    Weight distance) {

    const std::uint64_t distance_value =
        distance == INF
            ? 0xffffffffffffffffULL
            : static_cast<std::uint64_t>(distance);

    std::uint64_t value =
        (static_cast<std::uint64_t>(source) << 32) ^
        static_cast<std::uint64_t>(target);

    value ^= device_mix_query_value(distance_value);
    return device_mix_query_value(value);
}

__global__ void g32_initialize_distance_kernel(
    Weight* distance,
    std::uint64_t restricted_vertex_count) {

    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t count =
        restricted_vertex_count * static_cast<std::uint64_t>(G32_SOURCE_LANES);

    if (index < count) {
        distance[index] = INF;
    }
}

__global__ void g32_scatter_seed_kernel(
    const G32SeedRecord* seeds,
    std::uint64_t seed_count,
    Weight* distance) {

    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= seed_count) {
        return;
    }

    const G32SeedRecord seed = seeds[index];
    distance[
        static_cast<std::uint64_t>(seed.local_vertex) * G32_SOURCE_LANES +
        seed.lane] = seed.distance;
}

__global__ void g32_level_pull_kernel(
    const Vertex* level_vertices,
    std::uint64_t level_vertex_count,
    const Offset* incoming_first,
    const Vertex* incoming_tail,
    const Weight* incoming_weight,
    std::uint32_t active_source_count,
    Weight* distance) {

    const std::uint64_t global_thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t warp_index = global_thread / G32_SOURCE_LANES;
    const unsigned lane =
        static_cast<unsigned>(global_thread & (G32_SOURCE_LANES - 1));

    if (warp_index >= level_vertex_count || lane >= active_source_count) {
        return;
    }

    const Vertex local_v = level_vertices[warp_index];
    const std::uint64_t output_index =
        static_cast<std::uint64_t>(local_v) * G32_SOURCE_LANES + lane;

    Weight best = distance[output_index];

    for (Offset e = incoming_first[local_v];
         e < incoming_first[static_cast<std::size_t>(local_v) + 1];
         ++e) {

        const Vertex local_u = incoming_tail[e];
        const Weight du = distance[
            static_cast<std::uint64_t>(local_u) * G32_SOURCE_LANES + lane];
        const Weight w = incoming_weight[e];

        if (du == INF || w >= INF || du > INF - w) {
            continue;
        }

        const Weight candidate = du + w;
        if (candidate < best) {
            best = candidate;
        }
    }

    distance[output_index] = best;
}

__global__ void g32_finalize_kernel(
    const Vertex* target_local_index,
    const Vertex* target_ids,
    std::uint64_t target_count,
    const Vertex* batch_sources,
    std::uint32_t active_source_count,
    const Weight* distance,
    Weight* batch_results,
    unsigned long long* source_checksums,
    unsigned long long* source_unreachable) {

    const unsigned lane = threadIdx.x;
    if (blockIdx.x != 0 || lane >= active_source_count) {
        return;
    }

    const Vertex source = batch_sources[lane];
    unsigned long long checksum = 0;
    unsigned long long unreachable = 0;
    const std::uint64_t result_row =
        static_cast<std::uint64_t>(lane) * target_count;

    for (std::uint64_t target_index = 0;
         target_index < target_count;
         ++target_index) {

        const Vertex local_target = target_local_index[target_index];
        const Weight result = distance[
            static_cast<std::uint64_t>(local_target) * G32_SOURCE_LANES + lane];
        const Vertex target = target_ids[target_index];

        checksum ^= device_distance_fingerprint(source, target, result);
        if (result == INF) {
            ++unreachable;
        }
        if (batch_results != nullptr) {
            batch_results[result_row + target_index] = result;
        }
    }

    source_checksums[lane] = checksum;
    source_unreachable[lane] = unreachable;
}

__global__ void rphast_rank_sweep_kernel(
    const Offset* upward_first,
    const Vertex* upward_head,
    const Weight* upward_weight,
    const std::uint32_t* rank,
    const Vertex* vertex_by_rank,
    std::uint64_t vertex_count,
    const Vertex* restricted_vertex,
    const Offset* restricted_first,
    const Vertex* restricted_head,
    const Weight* restricted_weight,
    std::uint64_t restricted_vertex_count,
    const Vertex* target_ids,
    std::uint64_t target_count,
    const Vertex* batch_sources,
    std::uint64_t batch_source_count,
    Weight* distance_workspace,
    Weight* batch_results,
    unsigned long long* source_checksums,
    unsigned long long* source_unreachable) {

    const std::uint64_t source_index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (source_index >= batch_source_count) {
        return;
    }

    Weight* distance =
        distance_workspace + source_index * vertex_count;

    for (std::uint64_t v = 0; v < vertex_count; ++v) {
        distance[v] = INF;
    }

    const Vertex source = batch_sources[source_index];
    distance[source] = 0;

    const std::uint64_t source_rank = rank[source];

    // Exact DAG shortest paths in G^up.
    for (std::uint64_t r = source_rank; r < vertex_count; ++r) {
        const Vertex u = vertex_by_rank[r];
        const Weight du = distance[u];
        if (du == INF) {
            continue;
        }

        for (Offset e = upward_first[u]; e < upward_first[u + 1]; ++e) {
            const Vertex v = upward_head[e];
            const Weight w = upward_weight[e];
            if (w >= INF || du > INF - w) {
                continue;
            }
            const Weight candidate = du + w;
            if (candidate < distance[v]) {
                distance[v] = candidate;
            }
        }
    }

    for (std::uint64_t local_u = 0;
         local_u < restricted_vertex_count;
         ++local_u) {

        const Vertex u = restricted_vertex[local_u];
        const Weight du = distance[u];
        if (du == INF) {
            continue;
        }

        for (Offset e = restricted_first[local_u];
             e < restricted_first[local_u + 1];
             ++e) {

            const Vertex local_v = restricted_head[e];
            const Vertex v = restricted_vertex[local_v];
            const Weight w = restricted_weight[e];
            if (w >= INF || du > INF - w) {
                continue;
            }
            const Weight candidate = du + w;
            if (candidate < distance[v]) {
                distance[v] = candidate;
            }
        }
    }

    unsigned long long checksum = 0;
    unsigned long long unreachable = 0;

    const std::uint64_t result_row = source_index * target_count;
    for (std::uint64_t target_index = 0;
         target_index < target_count;
         ++target_index) {

        const Vertex target = target_ids[target_index];
        const Weight result = distance[target];

        checksum ^=
            device_distance_fingerprint(source, target, result);
        if (result == INF) {
            ++unreachable;
        }
        if (batch_results != nullptr) {
            batch_results[result_row + target_index] = result;
        }
    }

    source_checksums[source_index] = checksum;
    source_unreachable[source_index] = unreachable;
}

static std::size_t choose_cuda_batch_size(
    std::size_t requested,
    std::size_t source_count,
    std::size_t vertex_count,
    std::size_t target_count,
    bool quiet) {

    if (requested != 0) {
        return std::min(requested, source_count);
    }

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    cuda_check(
        cudaMemGetInfo(&free_bytes, &total_bytes),
        "cudaMemGetInfo");

    (void)total_bytes;

    const std::uint64_t per_source =
        static_cast<std::uint64_t>(vertex_count) * sizeof(Weight) +
        sizeof(Vertex) +
        2 * sizeof(unsigned long long) +
        (quiet
            ? 0
            : static_cast<std::uint64_t>(target_count) * sizeof(Weight));

    if (per_source == 0) {
        return 1;
    }

    const std::uint64_t budget =
        static_cast<std::uint64_t>(free_bytes) * 60ULL / 100ULL;

    std::uint64_t batch = budget / per_source;
    if (batch == 0) {
        batch = 1;
    }

    batch = std::min<std::uint64_t>(
        batch,
        static_cast<std::uint64_t>(source_count));

    return static_cast<std::size_t>(batch);
}

static void run_many_to_many_cuda(
    const CHQueryGraph& ch,
    const RestrictedDownward& restricted,
    const std::vector<Vertex>& sources,
    const std::vector<Vertex>& targets,
    bool quiet,
    int device_id,
    std::size_t requested_batch_sources,
    double selection_seconds) {

    using Clock = std::chrono::steady_clock;

    if (sources.empty() || targets.empty()) {
        throw std::runtime_error(
            "Source and target sets must both be non-empty");
    }

    if (sources.size() >
        std::numeric_limits<std::uint64_t>::max() / targets.size()) {
        throw std::runtime_error(
            "Number of source-target pairs overflows uint64_t");
    }

    const std::uint64_t pair_count =
        static_cast<std::uint64_t>(sources.size()) *
        static_cast<std::uint64_t>(targets.size());

    auto elapsed_seconds = [](Clock::time_point begin, Clock::time_point end) {
        return std::chrono::duration<double>(end - begin).count();
    };
    auto mib = [](std::uint64_t bytes) {
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    };
    auto checked_bytes = [](std::size_t count, std::size_t element_size,
                            const char* description) -> std::uint64_t {
        if (element_size != 0 &&
            count > std::numeric_limits<std::uint64_t>::max() / element_size) {
            throw std::runtime_error(
                std::string("Byte count overflows uint64_t for ") + description);
        }
        return static_cast<std::uint64_t>(count) * element_size;
    };

    const auto runtime_begin = Clock::now();
    cuda_check(cudaSetDevice(device_id), "cudaSetDevice");
    cudaDeviceProp properties{};
    cuda_check(
        cudaGetDeviceProperties(&properties, device_id),
        "cudaGetDeviceProperties");
    cuda_check(cudaFree(nullptr), "CUDA runtime initialization");
    const auto runtime_end = Clock::now();
    const double runtime_seconds = elapsed_seconds(runtime_begin, runtime_end);

    const std::size_t n = ch.upward.num_vertices();

    const std::uint64_t reusable_ch_bytes =
        checked_bytes(ch.upward.first.size(), sizeof(Offset), "upward first") +
        checked_bytes(ch.upward.head.size(), sizeof(Vertex), "upward head") +
        checked_bytes(ch.upward.weight.size(), sizeof(Weight), "upward weight") +
        checked_bytes(ch.rank.size(), sizeof(std::uint32_t), "rank") +
        checked_bytes(ch.vertex_by_rank.size(), sizeof(Vertex), "vertex_by_rank");

    const std::uint64_t restricted_bytes =
        checked_bytes(restricted.vertex.size(), sizeof(Vertex), "restricted vertex") +
        checked_bytes(restricted.first.size(), sizeof(Offset), "restricted first") +
        checked_bytes(restricted.head.size(), sizeof(Vertex), "restricted head") +
        checked_bytes(restricted.weight.size(), sizeof(Weight), "restricted weight") +
        checked_bytes(targets.size(), sizeof(Vertex), "targets");

    const auto static_alloc_begin = Clock::now();
    CudaBuffer<Offset> d_upward_first(ch.upward.first.size());
    CudaBuffer<Vertex> d_upward_head(ch.upward.head.size());
    CudaBuffer<Weight> d_upward_weight(ch.upward.weight.size());
    CudaBuffer<std::uint32_t> d_rank(ch.rank.size());
    CudaBuffer<Vertex> d_vertex_by_rank(ch.vertex_by_rank.size());

    CudaBuffer<Vertex> d_restricted_vertex(restricted.vertex.size());
    CudaBuffer<Offset> d_restricted_first(restricted.first.size());
    CudaBuffer<Vertex> d_restricted_head(restricted.head.size());
    CudaBuffer<Weight> d_restricted_weight(restricted.weight.size());
    CudaBuffer<Vertex> d_targets(targets.size());
    const auto static_alloc_end = Clock::now();
    const double static_alloc_seconds =
        elapsed_seconds(static_alloc_begin, static_alloc_end);

    const auto reusable_h2d_begin = Clock::now();
    d_upward_first.upload(ch.upward.first);
    d_upward_head.upload(ch.upward.head);
    d_upward_weight.upload(ch.upward.weight);
    d_rank.upload(ch.rank);
    d_vertex_by_rank.upload(ch.vertex_by_rank);
    const auto reusable_h2d_end = Clock::now();
    const double reusable_h2d_seconds =
        elapsed_seconds(reusable_h2d_begin, reusable_h2d_end);

    const auto restricted_h2d_begin = Clock::now();
    d_restricted_vertex.upload(restricted.vertex);
    d_restricted_first.upload(restricted.first);
    d_restricted_head.upload(restricted.head);
    d_restricted_weight.upload(restricted.weight);
    d_targets.upload(targets);
    const auto restricted_h2d_end = Clock::now();
    const double restricted_h2d_seconds =
        elapsed_seconds(restricted_h2d_begin, restricted_h2d_end);

    const auto planning_begin = Clock::now();
    const std::size_t batch_sources = choose_cuda_batch_size(
        requested_batch_sources,
        sources.size(),
        n,
        targets.size(),
        quiet);
    const auto planning_end = Clock::now();
    const double planning_seconds = elapsed_seconds(planning_begin, planning_end);

    if (n != 0 &&
        batch_sources >
            std::numeric_limits<std::size_t>::max() / n) {
        throw std::runtime_error("CUDA distance workspace size overflows size_t");
    }

    const std::size_t workspace_elements = batch_sources * n;

    const auto batch_alloc_begin = Clock::now();
    CudaBuffer<Weight> d_distance(workspace_elements);
    CudaBuffer<Vertex> d_batch_sources(batch_sources);
    CudaBuffer<unsigned long long> d_checksums(batch_sources);
    CudaBuffer<unsigned long long> d_unreachable(batch_sources);

    CudaBuffer<Weight> d_batch_results;
    if (!quiet) {
        if (targets.size() != 0 &&
            batch_sources >
                std::numeric_limits<std::size_t>::max() / targets.size()) {
            throw std::runtime_error("CUDA result batch size overflows size_t");
        }
        d_batch_results.allocate(batch_sources * targets.size());
    }
    const auto batch_alloc_end = Clock::now();
    const double batch_alloc_seconds =
        elapsed_seconds(batch_alloc_begin, batch_alloc_end);

    const auto host_prepare_begin = Clock::now();
    std::vector<Weight> host_results;
    if (!quiet) {
        if (pair_count >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error(
                "Too many results to materialize; use --quiet");
        }
        host_results.resize(static_cast<std::size_t>(pair_count), INF);
    }

    std::vector<unsigned long long> batch_checksums(batch_sources, 0);
    std::vector<unsigned long long> batch_unreachable(batch_sources, 0);
    const auto host_prepare_end = Clock::now();
    const double host_prepare_seconds =
        elapsed_seconds(host_prepare_begin, host_prepare_end);

    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    cuda_check(cudaEventCreate(&start_event), "cudaEventCreate start");
    cuda_check(cudaEventCreate(&stop_event), "cudaEventCreate stop");

    double source_h2d_seconds = 0.0;
    double kernel_seconds = 0.0;
    double stats_d2h_seconds = 0.0;
    double result_d2h_seconds = 0.0;
    double host_reduce_seconds = 0.0;
    std::uint64_t source_h2d_bytes = 0;
    std::uint64_t stats_d2h_bytes = 0;
    std::uint64_t result_d2h_bytes = 0;
    std::uint64_t checksum = 0;
    std::uint64_t unreachable = 0;
    std::uint64_t batch_count = 0;

    std::cerr
        << "RPHAST CUDA many-to-many: "
        << sources.size() << " sources x "
        << targets.size() << " targets = "
        << pair_count << " pairs\n"
        << "CUDA device " << device_id << ": "
        << properties.name << "\n"
        << "CUDA logic: one thread/source, upward rank sweep + restricted downward sweep\n"
        << "CUDA source batch size: " << batch_sources << "\n"
        << "dense distance workspace: "
        << std::fixed << std::setprecision(2)
        << (static_cast<double>(workspace_elements) * sizeof(Weight) /
            (1024.0 * 1024.0))
        << " MiB\n";

    constexpr unsigned threads_per_block = 128;

    const auto pipeline_begin = Clock::now();

    for (std::size_t source_begin = 0;
         source_begin < sources.size();
         source_begin += batch_sources) {

        ++batch_count;
        const std::size_t current_batch =
            std::min(batch_sources, sources.size() - source_begin);

        const auto source_h2d_begin = Clock::now();
        d_batch_sources.upload_slice(
            sources.data() + source_begin,
            current_batch);
        const auto source_h2d_end = Clock::now();
        source_h2d_seconds +=
            elapsed_seconds(source_h2d_begin, source_h2d_end);
        source_h2d_bytes +=
            checked_bytes(current_batch, sizeof(Vertex), "batch source IDs");

        const unsigned block_count =
            static_cast<unsigned>(
                (current_batch + threads_per_block - 1) /
                threads_per_block);

        cuda_check(cudaEventRecord(start_event), "cudaEventRecord start");

        rphast_rank_sweep_kernel<<<block_count, threads_per_block>>>(
            d_upward_first.data(),
            d_upward_head.data(),
            d_upward_weight.data(),
            d_rank.data(),
            d_vertex_by_rank.data(),
            static_cast<std::uint64_t>(n),
            d_restricted_vertex.data(),
            d_restricted_first.data(),
            d_restricted_head.data(),
            d_restricted_weight.data(),
            static_cast<std::uint64_t>(restricted.vertex.size()),
            d_targets.data(),
            static_cast<std::uint64_t>(targets.size()),
            d_batch_sources.data(),
            static_cast<std::uint64_t>(current_batch),
            d_distance.data(),
            quiet ? nullptr : d_batch_results.data(),
            d_checksums.data(),
            d_unreachable.data());

        cuda_check(cudaGetLastError(), "RPHAST CUDA kernel launch");
        cuda_check(cudaEventRecord(stop_event), "cudaEventRecord stop");
        cuda_check(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop");

        float milliseconds = 0.0f;
        cuda_check(
            cudaEventElapsedTime(&milliseconds, start_event, stop_event),
            "cudaEventElapsedTime");
        kernel_seconds += static_cast<double>(milliseconds) / 1000.0;

        const auto stats_d2h_begin = Clock::now();
        d_checksums.download_slice(batch_checksums.data(), current_batch);
        d_unreachable.download_slice(batch_unreachable.data(), current_batch);
        const auto stats_d2h_end = Clock::now();
        stats_d2h_seconds +=
            elapsed_seconds(stats_d2h_begin, stats_d2h_end);
        stats_d2h_bytes +=
            checked_bytes(current_batch, sizeof(unsigned long long), "checksums") +
            checked_bytes(current_batch, sizeof(unsigned long long), "unreachable counts");

        const auto reduce_begin = Clock::now();
        for (std::size_t i = 0; i < current_batch; ++i) {
            checksum ^= static_cast<std::uint64_t>(batch_checksums[i]);
            unreachable += static_cast<std::uint64_t>(batch_unreachable[i]);
        }
        const auto reduce_end = Clock::now();
        host_reduce_seconds += elapsed_seconds(reduce_begin, reduce_end);

        if (!quiet) {
            const std::size_t batch_result_count =
                current_batch * targets.size();
            const auto result_d2h_begin = Clock::now();
            d_batch_results.download_slice(
                host_results.data() + source_begin * targets.size(),
                batch_result_count);
            const auto result_d2h_end = Clock::now();
            result_d2h_seconds +=
                elapsed_seconds(result_d2h_begin, result_d2h_end);
            result_d2h_bytes +=
                checked_bytes(batch_result_count, sizeof(Weight), "distance results");
        }
    }

    const auto pipeline_end = Clock::now();
    const double pipeline_seconds = elapsed_seconds(pipeline_begin, pipeline_end);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    const double all_h2d_seconds =
        reusable_h2d_seconds + restricted_h2d_seconds + source_h2d_seconds;
    const std::uint64_t all_h2d_bytes =
        reusable_ch_bytes + restricted_bytes + source_h2d_bytes;
    const double all_d2h_seconds = stats_d2h_seconds + result_d2h_seconds;
    const std::uint64_t all_d2h_bytes = stats_d2h_bytes + result_d2h_bytes;

    const double resident_ch_seconds =
        selection_seconds + restricted_h2d_seconds + pipeline_seconds;
    const double cold_device_seconds =
        reusable_h2d_seconds + resident_ch_seconds;
    const double average_kernel_ns =
        kernel_seconds * 1.0e9 / static_cast<double>(pair_count);
    const double average_pipeline_ns =
        pipeline_seconds * 1.0e9 / static_cast<double>(pair_count);

    std::cerr
        << std::fixed << std::setprecision(6)
        << "CUDA timing breakdown\n"
        << "  runtime/device initialization (excluded): "
        << runtime_seconds << " s\n"
        << "  device allocation, static buffers (excluded): "
        << static_alloc_seconds << " s\n"
        << "  batch-size planning (excluded): "
        << planning_seconds << " s\n"
        << "  device allocation, batch workspace (excluded): "
        << batch_alloc_seconds << " s\n"
        << "  host result-buffer preparation (excluded): "
        << host_prepare_seconds << " s\n"
        << "Host -> device transfers\n"
        << "  reusable CH upward/rank data: "
        << reusable_h2d_seconds << " s ("
        << std::setprecision(2) << mib(reusable_ch_bytes) << " MiB)\n"
        << std::setprecision(6)
        << "  restricted graph + targets: "
        << restricted_h2d_seconds << " s ("
        << std::setprecision(2) << mib(restricted_bytes) << " MiB)\n"
        << std::setprecision(6)
        << "  per-batch source IDs: "
        << source_h2d_seconds << " s ("
        << std::setprecision(2) << mib(source_h2d_bytes) << " MiB)\n"
        << std::setprecision(6)
        << "  all H2D transfers: "
        << all_h2d_seconds << " s ("
        << std::setprecision(2) << mib(all_h2d_bytes) << " MiB)\n"
        << std::setprecision(6)
        << "Device/query stages\n"
        << "  CUDA kernel execution: " << kernel_seconds << " s\n"
        << "  checksum/unreachable D2H: "
        << stats_d2h_seconds << " s ("
        << std::setprecision(2) << mib(stats_d2h_bytes) << " MiB)\n"
        << std::setprecision(6)
        << "  distance-result D2H: "
        << result_d2h_seconds << " s ("
        << std::setprecision(2) << mib(result_d2h_bytes) << " MiB)"
        << (quiet ? " [quiet: disabled]" : "") << "\n"
        << std::setprecision(6)
        << "  all D2H transfers: "
        << all_d2h_seconds << " s ("
        << std::setprecision(2) << mib(all_d2h_bytes) << " MiB)\n"
        << std::setprecision(6)
        << "  host checksum reduction: " << host_reduce_seconds << " s\n"
        << "  batched query pipeline wall time: " << pipeline_seconds << " s\n"
        << "  CUDA batches executed: " << batch_count << "\n"
        << "Benchmark aggregates (CH file I/O excluded)\n"
        << "  target selection on CPU: " << selection_seconds << " s\n"
        << "  pure CUDA kernel time: " << kernel_seconds << " s\n"
        << "  query pipeline (all graph data resident): "
        << pipeline_seconds << " s\n"
        << "  resident-CH end-to-end: " << resident_ch_seconds << " s\n"
        << "    = selection + restricted H2D + pipeline\n"
        << "  cold-device end-to-end: " << cold_device_seconds << " s\n"
        << "    = resident-CH total + reusable CH H2D\n"
        << std::setprecision(2)
        << "  average pure kernel time: " << average_kernel_ns << " ns/pair\n"
        << "  average resident-pipeline time: " << average_pipeline_ns << " ns/pair\n"
        << "pairs completed: " << pair_count
        << ", unreachable: " << unreachable
        << ", distance checksum: " << checksum << "\n";

    if (!quiet) {
        for (std::size_t source_index = 0;
             source_index < sources.size();
             ++source_index) {
            for (std::size_t target_index = 0;
                 target_index < targets.size();
                 ++target_index) {

                print_distance_result(
                    sources[source_index],
                    targets[target_index],
                    host_results[source_index * targets.size() + target_index]);
            }
        }
    }
}


static void run_many_to_many_cuda_g32(
    const CHQueryGraph& ch,
    const RestrictedDownward& restricted,
    const std::vector<Vertex>& sources,
    const std::vector<Vertex>& targets,
    bool quiet,
    int device_id,
    std::size_t requested_batch_sources,
    int requested_threads,
    double selection_seconds) {

    using Clock = std::chrono::steady_clock;

    if (sources.empty() || targets.empty()) {
        throw std::runtime_error(
            "Source and target sets must both be non-empty");
    }
    if (requested_threads <= 0) {
        throw std::runtime_error("Thread count must be positive");
    }
#ifndef _OPENMP
    if (requested_threads != 1) {
        throw std::runtime_error(
            "cuda-g32 uses CPU upward searches and this executable was built "
            "without OpenMP; use --threads 1");
    }
#endif
    if (requested_batch_sources > G32_SOURCE_LANES) {
        throw std::runtime_error(
            "--gpu-batch-sources must be <= 32 for --backend cuda-g32");
    }

    if (sources.size() >
        std::numeric_limits<std::uint64_t>::max() / targets.size()) {
        throw std::runtime_error(
            "Number of source-target pairs overflows uint64_t");
    }

    const std::uint64_t pair_count =
        static_cast<std::uint64_t>(sources.size()) *
        static_cast<std::uint64_t>(targets.size());
    const std::size_t batch_sources =
        requested_batch_sources == 0
            ? static_cast<std::size_t>(G32_SOURCE_LANES)
            : requested_batch_sources;

    auto elapsed_seconds = [](Clock::time_point begin, Clock::time_point end) {
        return std::chrono::duration<double>(end - begin).count();
    };
    auto mib = [](std::uint64_t bytes) {
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    };
    auto checked_bytes = [](std::size_t count, std::size_t element_size,
                            const char* description) -> std::uint64_t {
        if (element_size != 0 &&
            count > std::numeric_limits<std::uint64_t>::max() / element_size) {
            throw std::runtime_error(
                std::string("Byte count overflows uint64_t for ") + description);
        }
        return static_cast<std::uint64_t>(count) * element_size;
    };

    const auto layout_begin = Clock::now();
    G32RestrictedLayout layout = build_g32_restricted_layout(
        restricted,
        ch.upward.num_vertices());
    const auto layout_end = Clock::now();
    const double layout_seconds = elapsed_seconds(layout_begin, layout_end);

    int actual_threads = 1;
#ifdef _OPENMP
    omp_set_dynamic(0);
#pragma omp parallel num_threads(requested_threads)
    {
#pragma omp single
        actual_threads = omp_get_num_threads();
    }
#endif

    std::vector<std::vector<G32UpwardSeed>> source_seeds(sources.size());
    const auto upward_begin = Clock::now();

#ifdef _OPENMP
#pragma omp parallel num_threads(requested_threads)
    {
        G32UpwardSeedWorkspace workspace(ch.upward.num_vertices());
#pragma omp for schedule(static)
        for (std::int64_t source_index = 0;
             source_index < static_cast<std::int64_t>(sources.size());
             ++source_index) {
            workspace.run(
                ch,
                layout.restricted_index,
                sources[static_cast<std::size_t>(source_index)],
                source_seeds[static_cast<std::size_t>(source_index)]);
        }
    }
#else
    {
        G32UpwardSeedWorkspace workspace(ch.upward.num_vertices());
        for (std::size_t source_index = 0;
             source_index < sources.size();
             ++source_index) {
            workspace.run(
                ch,
                layout.restricted_index,
                sources[source_index],
                source_seeds[source_index]);
        }
    }
#endif

    const auto upward_end = Clock::now();
    const double upward_seconds = elapsed_seconds(upward_begin, upward_end);

    const auto seed_plan_begin = Clock::now();
    std::uint64_t total_seed_count = 0;
    std::size_t maximum_batch_seed_count = 0;

    for (std::size_t source_begin = 0;
         source_begin < sources.size();
         source_begin += batch_sources) {

        const std::size_t current_batch =
            std::min(batch_sources, sources.size() - source_begin);
        std::size_t batch_seed_count = 0;

        for (std::size_t lane = 0; lane < current_batch; ++lane) {
            const std::size_t count = source_seeds[source_begin + lane].size();
            if (batch_seed_count >
                std::numeric_limits<std::size_t>::max() - count) {
                throw std::runtime_error("G32 batch seed count overflows size_t");
            }
            batch_seed_count += count;

            if (total_seed_count >
                std::numeric_limits<std::uint64_t>::max() - count) {
                throw std::runtime_error("G32 total seed count overflows uint64_t");
            }
            total_seed_count += static_cast<std::uint64_t>(count);
        }
        maximum_batch_seed_count =
            std::max(maximum_batch_seed_count, batch_seed_count);
    }

    std::vector<Vertex>().swap(layout.restricted_index);
    const auto seed_plan_end = Clock::now();
    const double seed_plan_seconds =
        elapsed_seconds(seed_plan_begin, seed_plan_end);

    const auto runtime_begin = Clock::now();
    cuda_check(cudaSetDevice(device_id), "cudaSetDevice");
    cudaDeviceProp properties{};
    cuda_check(
        cudaGetDeviceProperties(&properties, device_id),
        "cudaGetDeviceProperties");
    cuda_check(cudaFree(nullptr), "CUDA runtime initialization");
    const auto runtime_end = Clock::now();
    const double runtime_seconds = elapsed_seconds(runtime_begin, runtime_end);

    const std::size_t restricted_vertex_count = restricted.vertex.size();
    if (restricted_vertex_count >
        std::numeric_limits<std::size_t>::max() / G32_SOURCE_LANES) {
        throw std::runtime_error("G32 distance workspace size overflows size_t");
    }
    const std::size_t distance_elements =
        restricted_vertex_count * G32_SOURCE_LANES;

    if (!quiet && targets.size() >
        std::numeric_limits<std::size_t>::max() / G32_SOURCE_LANES) {
        throw std::runtime_error("G32 result batch size overflows size_t");
    }

    const std::uint64_t restricted_gpu_bytes =
        checked_bytes(layout.incoming_first.size(), sizeof(Offset),
                      "G32 incoming first") +
        checked_bytes(layout.incoming_tail.size(), sizeof(Vertex),
                      "G32 incoming tail") +
        checked_bytes(layout.incoming_weight.size(), sizeof(Weight),
                      "G32 incoming weight") +
        checked_bytes(layout.level_vertices.size(), sizeof(Vertex),
                      "G32 level vertices") +
        checked_bytes(layout.target_index.size(), sizeof(Vertex),
                      "G32 target local indices") +
        checked_bytes(targets.size(), sizeof(Vertex),
                      "G32 target IDs");

    const auto static_alloc_begin = Clock::now();
    CudaBuffer<Offset> d_incoming_first(layout.incoming_first.size());
    CudaBuffer<Vertex> d_incoming_tail(layout.incoming_tail.size());
    CudaBuffer<Weight> d_incoming_weight(layout.incoming_weight.size());
    CudaBuffer<Vertex> d_level_vertices(layout.level_vertices.size());
    CudaBuffer<Vertex> d_target_index(layout.target_index.size());
    CudaBuffer<Vertex> d_target_ids(targets.size());
    CudaBuffer<Weight> d_distance(distance_elements);
    CudaBuffer<G32SeedRecord> d_seed_records(
        std::max<std::size_t>(maximum_batch_seed_count, 1));
    CudaBuffer<Vertex> d_batch_sources(G32_SOURCE_LANES);
    CudaBuffer<unsigned long long> d_checksums(G32_SOURCE_LANES);
    CudaBuffer<unsigned long long> d_unreachable(G32_SOURCE_LANES);
    CudaBuffer<Weight> d_batch_results;
    if (!quiet) {
        d_batch_results.allocate(
            static_cast<std::size_t>(G32_SOURCE_LANES) * targets.size());
    }
    const auto static_alloc_end = Clock::now();
    const double allocation_seconds =
        elapsed_seconds(static_alloc_begin, static_alloc_end);

    const auto restricted_h2d_begin = Clock::now();
    d_incoming_first.upload(layout.incoming_first);
    d_incoming_tail.upload(layout.incoming_tail);
    d_incoming_weight.upload(layout.incoming_weight);
    d_level_vertices.upload(layout.level_vertices);
    d_target_index.upload(layout.target_index);
    d_target_ids.upload(targets);
    const auto restricted_h2d_end = Clock::now();
    const double restricted_h2d_seconds =
        elapsed_seconds(restricted_h2d_begin, restricted_h2d_end);

    const auto host_prepare_begin = Clock::now();
    std::vector<G32SeedRecord> batch_seed_records;
    batch_seed_records.reserve(maximum_batch_seed_count);
    std::vector<unsigned long long> batch_checksums(G32_SOURCE_LANES, 0);
    std::vector<unsigned long long> batch_unreachable(G32_SOURCE_LANES, 0);
    std::vector<Weight> host_results;
    if (!quiet) {
        if (pair_count >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error(
                "Too many results to materialize; use --quiet");
        }
        host_results.resize(static_cast<std::size_t>(pair_count), INF);
    }
    const auto host_prepare_end = Clock::now();
    const double host_prepare_seconds =
        elapsed_seconds(host_prepare_begin, host_prepare_end);

    cudaEvent_t event0 = nullptr;
    cudaEvent_t event1 = nullptr;
    cudaEvent_t event2 = nullptr;
    cudaEvent_t event3 = nullptr;
    cudaEvent_t event4 = nullptr;
    cuda_check(cudaEventCreate(&event0), "cudaEventCreate G32 event0");
    cuda_check(cudaEventCreate(&event1), "cudaEventCreate G32 event1");
    cuda_check(cudaEventCreate(&event2), "cudaEventCreate G32 event2");
    cuda_check(cudaEventCreate(&event3), "cudaEventCreate G32 event3");
    cuda_check(cudaEventCreate(&event4), "cudaEventCreate G32 event4");

    auto event_seconds = [](cudaEvent_t begin, cudaEvent_t end) {
        float milliseconds = 0.0f;
        cuda_check(
            cudaEventElapsedTime(&milliseconds, begin, end),
            "cudaEventElapsedTime G32");
        return static_cast<double>(milliseconds) / 1000.0;
    };

    auto checked_blocks = [&properties](
        std::uint64_t work_items,
        unsigned threads_per_block,
        const char* description) -> unsigned {

        if (work_items == 0) {
            return 0;
        }
        const std::uint64_t blocks =
            (work_items + threads_per_block - 1) / threads_per_block;
        if (blocks > static_cast<std::uint64_t>(properties.maxGridSize[0]) ||
            blocks > std::numeric_limits<unsigned>::max()) {
            throw std::runtime_error(
                std::string("CUDA grid is too large for ") + description);
        }
        return static_cast<unsigned>(blocks);
    };

    double seed_flatten_seconds = 0.0;
    double dynamic_h2d_seconds = 0.0;
    double initialize_kernel_seconds = 0.0;
    double scatter_kernel_seconds = 0.0;
    double downward_kernel_seconds = 0.0;
    double finalize_kernel_seconds = 0.0;
    double device_sequence_seconds = 0.0;
    double stats_d2h_seconds = 0.0;
    double result_d2h_seconds = 0.0;
    double host_reduce_seconds = 0.0;
    std::uint64_t dynamic_h2d_bytes = 0;
    std::uint64_t stats_d2h_bytes = 0;
    std::uint64_t result_d2h_bytes = 0;
    std::uint64_t checksum = 0;
    std::uint64_t unreachable = 0;
    std::uint64_t batch_count = 0;
    std::uint64_t level_kernel_launches = 0;

    constexpr unsigned generic_threads = 256;
    constexpr unsigned level_threads = 256;
    const std::uint64_t initialize_work =
        static_cast<std::uint64_t>(restricted_vertex_count) *
        G32_SOURCE_LANES;
    const unsigned initialize_blocks = checked_blocks(
        initialize_work,
        generic_threads,
        "G32 distance initialization");

    std::cerr
        << "G-RPHAST-32 CUDA many-to-many: "
        << sources.size() << " sources x "
        << targets.size() << " targets = "
        << pair_count << " pairs\n"
        << "CUDA device " << device_id << ": " << properties.name << "\n"
        << "CPU upward workers: " << actual_threads
#ifdef _OPENMP
        << " (OpenMP)\n"
#else
        << " (serial)\n"
#endif
        << "CUDA mapping: one warp/restricted vertex, one lane/source\n"
        << "source lanes per batch: " << batch_sources << " / 32\n"
        << "restricted dependency levels: "
        << layout.level_offsets.size() - 1
        << " (maximum derived level " << layout.maximum_level << ")\n"
        << "sparse upward seeds: " << total_seed_count
        << " total, maximum " << maximum_batch_seed_count
        << " in one batch\n"
        << "GPU distance workspace: "
        << std::fixed << std::setprecision(2)
        << (static_cast<double>(distance_elements) * sizeof(Weight) /
            (1024.0 * 1024.0))
        << " MiB\n";

    const auto pipeline_begin = Clock::now();

    for (std::size_t source_begin = 0;
         source_begin < sources.size();
         source_begin += batch_sources) {

        ++batch_count;
        const std::size_t current_batch =
            std::min(batch_sources, sources.size() - source_begin);

        const auto flatten_begin = Clock::now();
        batch_seed_records.clear();
        for (std::size_t lane = 0; lane < current_batch; ++lane) {
            for (const G32UpwardSeed& seed : source_seeds[source_begin + lane]) {
                batch_seed_records.push_back(G32SeedRecord{
                    seed.local_vertex,
                    static_cast<std::uint32_t>(lane),
                    seed.distance});
            }
        }
        const auto flatten_end = Clock::now();
        seed_flatten_seconds += elapsed_seconds(flatten_begin, flatten_end);

        const auto h2d_begin = Clock::now();
        d_batch_sources.upload_slice(
            sources.data() + source_begin,
            current_batch);
        if (!batch_seed_records.empty()) {
            d_seed_records.upload_slice(
                batch_seed_records.data(),
                batch_seed_records.size());
        }
        const auto h2d_end = Clock::now();
        dynamic_h2d_seconds += elapsed_seconds(h2d_begin, h2d_end);
        dynamic_h2d_bytes +=
            checked_bytes(current_batch, sizeof(Vertex), "G32 batch sources") +
            checked_bytes(batch_seed_records.size(), sizeof(G32SeedRecord),
                          "G32 sparse seeds");

        cuda_check(cudaEventRecord(event0), "cudaEventRecord G32 event0");

        g32_initialize_distance_kernel<<<initialize_blocks, generic_threads>>>(
            d_distance.data(),
            static_cast<std::uint64_t>(restricted_vertex_count));
        cuda_check(cudaGetLastError(), "G32 initialize kernel launch");
        cuda_check(cudaEventRecord(event1), "cudaEventRecord G32 event1");

        if (!batch_seed_records.empty()) {
            const unsigned scatter_blocks = checked_blocks(
                static_cast<std::uint64_t>(batch_seed_records.size()),
                generic_threads,
                "G32 seed scatter");
            g32_scatter_seed_kernel<<<scatter_blocks, generic_threads>>>(
                d_seed_records.data(),
                static_cast<std::uint64_t>(batch_seed_records.size()),
                d_distance.data());
            cuda_check(cudaGetLastError(), "G32 seed-scatter kernel launch");
        }
        cuda_check(cudaEventRecord(event2), "cudaEventRecord G32 event2");

        for (std::size_t bucket = 0;
             bucket + 1 < layout.level_offsets.size();
             ++bucket) {

            const Offset begin = layout.level_offsets[bucket];
            const Offset end = layout.level_offsets[bucket + 1];
            const std::uint64_t level_vertex_count =
                static_cast<std::uint64_t>(end - begin);
            if (level_vertex_count == 0) {
                continue;
            }

            const std::uint64_t thread_work =
                level_vertex_count * G32_SOURCE_LANES;
            const unsigned level_blocks = checked_blocks(
                thread_work,
                level_threads,
                "G32 level pull");

            g32_level_pull_kernel<<<level_blocks, level_threads>>>(
                d_level_vertices.data() + static_cast<std::size_t>(begin),
                level_vertex_count,
                d_incoming_first.data(),
                d_incoming_tail.data(),
                d_incoming_weight.data(),
                static_cast<std::uint32_t>(current_batch),
                d_distance.data());
            cuda_check(cudaGetLastError(), "G32 level-pull kernel launch");
            ++level_kernel_launches;
        }

        cuda_check(cudaEventRecord(event3), "cudaEventRecord G32 event3");

        g32_finalize_kernel<<<1, G32_SOURCE_LANES>>>(
            d_target_index.data(),
            d_target_ids.data(),
            static_cast<std::uint64_t>(targets.size()),
            d_batch_sources.data(),
            static_cast<std::uint32_t>(current_batch),
            d_distance.data(),
            quiet ? nullptr : d_batch_results.data(),
            d_checksums.data(),
            d_unreachable.data());
        cuda_check(cudaGetLastError(), "G32 finalize kernel launch");

        cuda_check(cudaEventRecord(event4), "cudaEventRecord G32 event4");
        cuda_check(cudaEventSynchronize(event4), "cudaEventSynchronize G32");

        initialize_kernel_seconds += event_seconds(event0, event1);
        scatter_kernel_seconds += event_seconds(event1, event2);
        downward_kernel_seconds += event_seconds(event2, event3);
        finalize_kernel_seconds += event_seconds(event3, event4);
        device_sequence_seconds += event_seconds(event0, event4);

        const auto stats_begin = Clock::now();
        d_checksums.download_slice(batch_checksums.data(), current_batch);
        d_unreachable.download_slice(batch_unreachable.data(), current_batch);
        const auto stats_end = Clock::now();
        stats_d2h_seconds += elapsed_seconds(stats_begin, stats_end);
        stats_d2h_bytes +=
            checked_bytes(current_batch, sizeof(unsigned long long),
                          "G32 checksums") +
            checked_bytes(current_batch, sizeof(unsigned long long),
                          "G32 unreachable counts");

        const auto reduce_begin = Clock::now();
        for (std::size_t lane = 0; lane < current_batch; ++lane) {
            checksum ^= static_cast<std::uint64_t>(batch_checksums[lane]);
            unreachable +=
                static_cast<std::uint64_t>(batch_unreachable[lane]);
        }
        const auto reduce_end = Clock::now();
        host_reduce_seconds += elapsed_seconds(reduce_begin, reduce_end);

        if (!quiet) {
            const std::size_t batch_result_count =
                current_batch * targets.size();
            const auto result_begin = Clock::now();
            d_batch_results.download_slice(
                host_results.data() + source_begin * targets.size(),
                batch_result_count);
            const auto result_end = Clock::now();
            result_d2h_seconds += elapsed_seconds(result_begin, result_end);
            result_d2h_bytes += checked_bytes(
                batch_result_count,
                sizeof(Weight),
                "G32 distance results");
        }
    }

    const auto pipeline_end = Clock::now();
    const double gpu_pipeline_seconds =
        elapsed_seconds(pipeline_begin, pipeline_end);

    cudaEventDestroy(event0);
    cudaEventDestroy(event1);
    cudaEventDestroy(event2);
    cudaEventDestroy(event3);
    cudaEventDestroy(event4);

    const double all_h2d_seconds =
        restricted_h2d_seconds + dynamic_h2d_seconds;
    const std::uint64_t all_h2d_bytes =
        restricted_gpu_bytes + dynamic_h2d_bytes;
    const double all_d2h_seconds =
        stats_d2h_seconds + result_d2h_seconds;
    const std::uint64_t all_d2h_bytes =
        stats_d2h_bytes + result_d2h_bytes;

    const double resident_target_query_seconds =
        upward_seconds + seed_plan_seconds + gpu_pipeline_seconds;
    const double resident_ch_end_to_end_seconds =
        selection_seconds + layout_seconds + restricted_h2d_seconds +
        resident_target_query_seconds;

    const double average_device_ns =
        device_sequence_seconds * 1.0e9 / static_cast<double>(pair_count);
    const double average_resident_query_ns =
        resident_target_query_seconds * 1.0e9 /
        static_cast<double>(pair_count);

    std::cerr
        << std::fixed << std::setprecision(6)
        << "G-RPHAST-32 timing breakdown\n"
        << "  CUDA runtime/device initialization (excluded): "
        << runtime_seconds << " s\n"
        << "  CUDA allocations (excluded): "
        << allocation_seconds << " s\n"
        << "  host result/batch-buffer preparation (excluded): "
        << host_prepare_seconds << " s\n"
        << "CPU/query preparation\n"
        << "  target selection (ordinary RPHAST): "
        << selection_seconds << " s\n"
        << "  G32 incoming-CSR/level layout build: "
        << layout_seconds << " s\n"
        << "  OpenMP upward CH searches + seed extraction: "
        << upward_seconds << " s\n"
        << "  seed-count planning: " << seed_plan_seconds << " s\n"
        << "  per-batch sparse-seed flattening: "
        << seed_flatten_seconds << " s\n"
        << "Host -> device transfers\n"
        << "  target-dependent restricted graph: "
        << restricted_h2d_seconds << " s ("
        << std::setprecision(2) << mib(restricted_gpu_bytes) << " MiB)\n"
        << std::setprecision(6)
        << "  per-batch source IDs + sparse seeds: "
        << dynamic_h2d_seconds << " s ("
        << std::setprecision(2) << mib(dynamic_h2d_bytes) << " MiB)\n"
        << std::setprecision(6)
        << "  all H2D transfers: " << all_h2d_seconds << " s ("
        << std::setprecision(2) << mib(all_h2d_bytes) << " MiB)\n"
        << std::setprecision(6)
        << "CUDA device stages\n"
        << "  distance initialization kernels: "
        << initialize_kernel_seconds << " s\n"
        << "  sparse-seed scatter kernels: "
        << scatter_kernel_seconds << " s\n"
        << "  level-by-level downward pull: "
        << downward_kernel_seconds << " s\n"
        << "  finalization/checksum kernels: "
        << finalize_kernel_seconds << " s\n"
        << "  complete CUDA device sequence: "
        << device_sequence_seconds << " s\n"
        << "  level kernel launches: " << level_kernel_launches << "\n"
        << "Device -> host / host finalization\n"
        << "  checksum/unreachable D2H: "
        << stats_d2h_seconds << " s ("
        << std::setprecision(2) << mib(stats_d2h_bytes) << " MiB)\n"
        << std::setprecision(6)
        << "  distance-result D2H: "
        << result_d2h_seconds << " s ("
        << std::setprecision(2) << mib(result_d2h_bytes) << " MiB)"
        << (quiet ? " [quiet: disabled]" : "") << "\n"
        << std::setprecision(6)
        << "  all D2H transfers: " << all_d2h_seconds << " s ("
        << std::setprecision(2) << mib(all_d2h_bytes) << " MiB)\n"
        << std::setprecision(6)
        << "  host checksum reduction: " << host_reduce_seconds << " s\n"
        << "  batched GPU pipeline wall time: "
        << gpu_pipeline_seconds << " s\n"
        << "Benchmark aggregates (CH file I/O excluded)\n"
        << "  pure CUDA device sequence: "
        << device_sequence_seconds << " s\n"
        << "  resident-target query: "
        << resident_target_query_seconds << " s\n"
        << "    = CPU upward searches + seed planning + batched GPU pipeline\n"
        << "  resident-CH end-to-end: "
        << resident_ch_end_to_end_seconds << " s\n"
        << "    = selection + G32 layout + restricted H2D + resident-target query\n"
        << "  CUDA batches executed: " << batch_count << "\n"
        << std::setprecision(2)
        << "  average pure device time: " << average_device_ns << " ns/pair\n"
        << "  average resident-target query time: "
        << average_resident_query_ns << " ns/pair\n"
        << "pairs completed: " << pair_count
        << ", unreachable: " << unreachable
        << ", distance checksum: " << checksum << "\n";

    if (!quiet) {
        for (std::size_t source_index = 0;
             source_index < sources.size();
             ++source_index) {
            for (std::size_t target_index = 0;
                 target_index < targets.size();
                 ++target_index) {
                print_distance_result(
                    sources[source_index],
                    targets[target_index],
                    host_results[source_index * targets.size() + target_index]);
            }
        }
    }
}


struct G32OptimizedQueryStats {
    std::uint64_t source_count = 0;
    std::uint64_t target_count = 0;
    std::uint64_t pair_count = 0;

    double runtime_seconds = 0.0;
    double selection_seconds = 0.0;
    double layout_seconds = 0.0;
    double allocation_seconds = 0.0;
    double seed_buffer_growth_seconds = 0.0;
    double restricted_h2d_seconds = 0.0;
    double upward_seconds = 0.0;
    double seed_flatten_seconds = 0.0;
    double dynamic_h2d_seconds = 0.0;
    double initialize_kernel_seconds = 0.0;
    double scatter_kernel_seconds = 0.0;
    double downward_kernel_seconds = 0.0;
    double finalize_kernel_seconds = 0.0;
    double device_sequence_seconds = 0.0;
    double stats_d2h_seconds = 0.0;
    double result_d2h_seconds = 0.0;
    double host_reduce_seconds = 0.0;
    double streaming_seconds = 0.0;
    double end_to_end_seconds = 0.0;

    std::uint64_t restricted_h2d_bytes = 0;
    std::uint64_t dynamic_h2d_bytes = 0;
    std::uint64_t stats_d2h_bytes = 0;
    std::uint64_t result_d2h_bytes = 0;
    std::uint64_t total_seed_count = 0;
    std::uint64_t checksum = 0;
    std::uint64_t unreachable = 0;
    std::uint64_t batch_count = 0;
    std::uint64_t level_kernel_launches = 0;

    std::uint64_t restricted_vertex_count = 0;
    std::uint64_t restricted_arc_count = 0;
    std::uint64_t restricted_level_count = 0;
    bool used_full_rank_scan = false;

    std::uint64_t distance_workspace_bytes = 0;
    std::uint64_t seed_buffer_peak_bytes = 0;
    std::uint64_t device_query_peak_bytes = 0;
    std::uint64_t device_free_before_bytes = 0;
    std::uint64_t host_layout_bytes = 0;
    std::uint64_t host_workspace_bytes_before = 0;
    std::uint64_t host_workspace_bytes_after = 0;
    std::uint64_t host_auxiliary_bytes = 0;
};

static G32OptimizedQueryStats run_many_to_many_cuda_g32_optimized(
    const CHQueryGraph& ch,
    G32OptimizedHostWorkspace& host_workspace,
    const std::vector<Vertex>& sources,
    const std::vector<Vertex>& targets,
    bool quiet,
    int device_id,
    std::size_t requested_batch_sources,
    int requested_threads,
    double reusable_host_workspace_init_seconds,
    bool print_report = true) {

    using Clock = std::chrono::steady_clock;
    if (sources.empty() || targets.empty()) {
        throw std::runtime_error("Source and target sets must both be non-empty");
    }
    if (requested_threads <= 0) {
        throw std::runtime_error("Thread count must be positive");
    }
#ifndef _OPENMP
    if (requested_threads != 1) {
        throw std::runtime_error(
            "cuda-g32-opt uses CPU upward searches; rebuild with OpenMP or use --threads 1");
    }
#endif
    if (requested_batch_sources > G32_SOURCE_LANES) {
        throw std::runtime_error(
            "--gpu-batch-sources must be <= 32 for --backend cuda-g32-opt");
    }
    if (sources.size() >
        std::numeric_limits<std::uint64_t>::max() / targets.size()) {
        throw std::runtime_error("Number of source-target pairs overflows uint64_t");
    }

    const std::uint64_t pair_count =
        static_cast<std::uint64_t>(sources.size()) *
        static_cast<std::uint64_t>(targets.size());
    const std::size_t batch_sources =
        requested_batch_sources == 0
            ? static_cast<std::size_t>(G32_SOURCE_LANES)
            : requested_batch_sources;

    auto elapsed_seconds = [](Clock::time_point begin, Clock::time_point end) {
        return std::chrono::duration<double>(end - begin).count();
    };
    auto mib = [](std::uint64_t bytes) {
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    };
    auto checked_bytes = [](std::size_t count, std::size_t element_size,
                            const char* description) -> std::uint64_t {
        if (element_size != 0 &&
            count > std::numeric_limits<std::uint64_t>::max() / element_size) {
            throw std::runtime_error(
                std::string("Byte count overflows uint64_t for ") + description);
        }
        return static_cast<std::uint64_t>(count) * element_size;
    };

    const auto runtime_begin = Clock::now();
    cuda_check(cudaSetDevice(device_id), "cudaSetDevice G32 opt");
    cudaDeviceProp properties{};
    cuda_check(
        cudaGetDeviceProperties(&properties, device_id),
        "cudaGetDeviceProperties G32 opt");
    cuda_check(cudaFree(nullptr), "CUDA runtime initialization G32 opt");
    const auto runtime_end = Clock::now();
    const double runtime_seconds = elapsed_seconds(runtime_begin, runtime_end);

    int actual_threads = 1;
#ifdef _OPENMP
    omp_set_dynamic(0);
#pragma omp parallel num_threads(requested_threads)
    {
#pragma omp single
        actual_threads = omp_get_num_threads();
    }
#endif
    if (static_cast<std::size_t>(actual_threads) >
        host_workspace.upward_workers.size()) {
        throw std::runtime_error("Persistent G32 CPU workspace has too few workers");
    }

    cudaEvent_t event0 = nullptr;
    cudaEvent_t event1 = nullptr;
    cudaEvent_t event2 = nullptr;
    cudaEvent_t event3 = nullptr;
    cudaEvent_t event4 = nullptr;
    cuda_check(cudaEventCreate(&event0), "cudaEventCreate G32 opt event0");
    cuda_check(cudaEventCreate(&event1), "cudaEventCreate G32 opt event1");
    cuda_check(cudaEventCreate(&event2), "cudaEventCreate G32 opt event2");
    cuda_check(cudaEventCreate(&event3), "cudaEventCreate G32 opt event3");
    cuda_check(cudaEventCreate(&event4), "cudaEventCreate G32 opt event4");

    const std::uint64_t host_workspace_bytes_before =
        host_workspace.capacity_bytes();

    std::size_t device_free_before = 0;
    std::size_t device_total_before = 0;
    cuda_check(
        cudaMemGetInfo(&device_free_before, &device_total_before),
        "cudaMemGetInfo G32 opt before query");

    const auto query_wall_begin = Clock::now();

    const auto selection_begin = Clock::now();
    host_workspace.selection.select(ch, targets);
    const auto selection_end = Clock::now();
    const double selection_seconds = elapsed_seconds(selection_begin, selection_end);

    const auto layout_begin = Clock::now();
    G32DirectLayout layout =
        build_g32_direct_layout(ch, targets, host_workspace.selection);
    const auto layout_end = Clock::now();
    const double layout_seconds = elapsed_seconds(layout_begin, layout_end);

    const std::size_t restricted_vertex_count = layout.restricted_vertex_count;
    if (restricted_vertex_count >
        std::numeric_limits<std::size_t>::max() / G32_SOURCE_LANES) {
        throw std::runtime_error("G32 distance workspace size overflows size_t");
    }
    const std::size_t distance_elements =
        restricted_vertex_count * G32_SOURCE_LANES;

    const std::uint64_t restricted_gpu_bytes =
        checked_bytes(layout.incoming_first.size(), sizeof(Offset),
                      "optimized G32 incoming first") +
        checked_bytes(layout.incoming_tail.size(), sizeof(Vertex),
                      "optimized G32 incoming tail") +
        checked_bytes(layout.incoming_weight.size(), sizeof(Weight),
                      "optimized G32 incoming weight") +
        checked_bytes(layout.level_vertices.size(), sizeof(Vertex),
                      "optimized G32 level vertices") +
        checked_bytes(layout.target_index.size(), sizeof(Vertex),
                      "optimized G32 target indices") +
        checked_bytes(targets.size(), sizeof(Vertex),
                      "optimized G32 target IDs");

    const std::uint64_t distance_workspace_bytes =
        checked_bytes(distance_elements, sizeof(Weight),
                      "optimized G32 distance workspace");
    std::uint64_t fixed_device_query_bytes = restricted_gpu_bytes;
    fixed_device_query_bytes += distance_workspace_bytes;
    fixed_device_query_bytes +=
        checked_bytes(G32_SOURCE_LANES, sizeof(Vertex), "optimized G32 batch source buffer") +
        checked_bytes(G32_SOURCE_LANES, sizeof(unsigned long long), "optimized G32 checksum buffer") +
        checked_bytes(G32_SOURCE_LANES, sizeof(unsigned long long), "optimized G32 unreachable buffer");
    if (!quiet) {
        fixed_device_query_bytes +=
            checked_bytes(static_cast<std::size_t>(G32_SOURCE_LANES) * targets.size(),
                          sizeof(Weight), "optimized G32 result buffer");
    }
    const std::uint64_t host_layout_bytes = layout.capacity_bytes();

    const auto allocation_begin = Clock::now();
    CudaBuffer<Offset> d_incoming_first(layout.incoming_first.size());
    CudaBuffer<Vertex> d_incoming_tail(layout.incoming_tail.size());
    CudaBuffer<Weight> d_incoming_weight(layout.incoming_weight.size());
    CudaBuffer<Vertex> d_level_vertices(layout.level_vertices.size());
    CudaBuffer<Vertex> d_target_index(layout.target_index.size());
    CudaBuffer<Vertex> d_target_ids(targets.size());
    CudaBuffer<Weight> d_distance(distance_elements);
    CudaBuffer<G32SeedRecord> d_seed_records;
    CudaBuffer<Vertex> d_batch_sources(G32_SOURCE_LANES);
    CudaBuffer<unsigned long long> d_checksums(G32_SOURCE_LANES);
    CudaBuffer<unsigned long long> d_unreachable(G32_SOURCE_LANES);
    CudaBuffer<Weight> d_batch_results;
    if (!quiet) {
        if (targets.size() >
            std::numeric_limits<std::size_t>::max() / G32_SOURCE_LANES) {
            throw std::runtime_error("G32 result batch size overflows size_t");
        }
        d_batch_results.allocate(
            static_cast<std::size_t>(G32_SOURCE_LANES) * targets.size());
    }
    const auto allocation_end = Clock::now();
    double allocation_seconds = elapsed_seconds(allocation_begin, allocation_end);
    double seed_buffer_growth_seconds = 0.0;

    const auto restricted_h2d_begin = Clock::now();
    d_incoming_first.upload(layout.incoming_first);
    d_incoming_tail.upload(layout.incoming_tail);
    d_incoming_weight.upload(layout.incoming_weight);
    d_level_vertices.upload(layout.level_vertices);
    d_target_index.upload(layout.target_index);
    d_target_ids.upload(targets);
    const auto restricted_h2d_end = Clock::now();
    const double restricted_h2d_seconds =
        elapsed_seconds(restricted_h2d_begin, restricted_h2d_end);

    std::vector<unsigned long long> batch_checksums(G32_SOURCE_LANES, 0);
    std::vector<unsigned long long> batch_unreachable(G32_SOURCE_LANES, 0);
    std::vector<Weight> host_results;
    if (!quiet) {
        if (pair_count > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("Too many results to materialize; use --quiet");
        }
        host_results.resize(static_cast<std::size_t>(pair_count), INF);
    }
    const std::uint64_t host_auxiliary_bytes =
        static_cast<std::uint64_t>(batch_checksums.capacity()) * sizeof(unsigned long long) +
        static_cast<std::uint64_t>(batch_unreachable.capacity()) * sizeof(unsigned long long) +
        static_cast<std::uint64_t>(host_results.capacity()) * sizeof(Weight);

    auto event_seconds = [](cudaEvent_t begin, cudaEvent_t end) {
        float milliseconds = 0.0f;
        cuda_check(
            cudaEventElapsedTime(&milliseconds, begin, end),
            "cudaEventElapsedTime G32 opt");
        return static_cast<double>(milliseconds) / 1000.0;
    };

    auto checked_blocks = [&properties](
        std::uint64_t work_items,
        unsigned threads_per_block,
        const char* description) -> unsigned {
        if (work_items == 0) {
            return 0;
        }
        const std::uint64_t blocks =
            (work_items + threads_per_block - 1) / threads_per_block;
        if (blocks > static_cast<std::uint64_t>(properties.maxGridSize[0]) ||
            blocks > std::numeric_limits<unsigned>::max()) {
            throw std::runtime_error(
                std::string("CUDA grid is too large for ") + description);
        }
        return static_cast<unsigned>(blocks);
    };

    constexpr unsigned generic_threads = 256;
    constexpr unsigned level_threads = 256;
    const std::uint64_t initialize_work =
        static_cast<std::uint64_t>(restricted_vertex_count) * G32_SOURCE_LANES;
    const unsigned initialize_blocks = checked_blocks(
        initialize_work, generic_threads, "optimized G32 distance initialization");

    double upward_seconds = 0.0;
    double seed_flatten_seconds = 0.0;
    double dynamic_h2d_seconds = 0.0;
    double initialize_kernel_seconds = 0.0;
    double scatter_kernel_seconds = 0.0;
    double downward_kernel_seconds = 0.0;
    double finalize_kernel_seconds = 0.0;
    double device_sequence_seconds = 0.0;
    double stats_d2h_seconds = 0.0;
    double result_d2h_seconds = 0.0;
    double host_reduce_seconds = 0.0;
    std::uint64_t dynamic_h2d_bytes = 0;
    std::uint64_t stats_d2h_bytes = 0;
    std::uint64_t result_d2h_bytes = 0;
    std::uint64_t total_seed_count = 0;
    std::uint64_t checksum = 0;
    std::uint64_t unreachable = 0;
    std::uint64_t batch_count = 0;
    std::uint64_t level_kernel_launches = 0;

    const auto streaming_begin = Clock::now();

    for (std::size_t source_begin = 0;
         source_begin < sources.size();
         source_begin += batch_sources) {

        ++batch_count;
        const std::size_t current_batch =
            std::min(batch_sources, sources.size() - source_begin);

        const auto upward_begin = Clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(requested_threads)
        for (std::int64_t lane_i = 0;
             lane_i < static_cast<std::int64_t>(current_batch);
             ++lane_i) {
            const int tid = omp_get_thread_num();
            const std::size_t lane = static_cast<std::size_t>(lane_i);
            host_workspace.upward_workers[static_cast<std::size_t>(tid)]->run(
                ch,
                host_workspace.selection,
                sources[source_begin + lane],
                host_workspace.lane_seeds[lane]);
        }
#else
        for (std::size_t lane = 0; lane < current_batch; ++lane) {
            host_workspace.upward_workers[0]->run(
                ch,
                host_workspace.selection,
                sources[source_begin + lane],
                host_workspace.lane_seeds[lane]);
        }
#endif
        const auto upward_end = Clock::now();
        upward_seconds += elapsed_seconds(upward_begin, upward_end);

        const auto flatten_begin = Clock::now();
        host_workspace.flat_seeds.clear();
        std::size_t batch_seed_count = 0;
        for (std::size_t lane = 0; lane < current_batch; ++lane) {
            if (batch_seed_count > std::numeric_limits<std::size_t>::max() -
                                       host_workspace.lane_seeds[lane].size()) {
                throw std::runtime_error("Optimized G32 batch seed count overflow");
            }
            batch_seed_count += host_workspace.lane_seeds[lane].size();
        }
        if (host_workspace.flat_seeds.capacity() < batch_seed_count) {
            host_workspace.flat_seeds.reserve(batch_seed_count);
        }
        for (std::size_t lane = 0; lane < current_batch; ++lane) {
            for (const G32UpwardSeed& seed : host_workspace.lane_seeds[lane]) {
                host_workspace.flat_seeds.push_back(G32SeedRecord{
                    seed.local_vertex,
                    static_cast<std::uint32_t>(lane),
                    seed.distance});
            }
        }
        total_seed_count += static_cast<std::uint64_t>(batch_seed_count);
        const auto flatten_end = Clock::now();
        seed_flatten_seconds += elapsed_seconds(flatten_begin, flatten_end);

        if (batch_seed_count > d_seed_records.size()) {
            std::size_t new_capacity = std::max<std::size_t>(1, d_seed_records.size());
            while (new_capacity < batch_seed_count) {
                const std::size_t grown =
                    new_capacity > std::numeric_limits<std::size_t>::max() / 2
                        ? batch_seed_count
                        : new_capacity * 2;
                if (grown <= new_capacity) {
                    new_capacity = batch_seed_count;
                    break;
                }
                new_capacity = grown;
            }
            const auto grow_begin = Clock::now();
            d_seed_records.allocate(new_capacity);
            const auto grow_end = Clock::now();
            const double grow_seconds = elapsed_seconds(grow_begin, grow_end);
            seed_buffer_growth_seconds += grow_seconds;
            allocation_seconds += grow_seconds;
        }

        const auto h2d_begin = Clock::now();
        d_batch_sources.upload_slice(
            sources.data() + source_begin, current_batch);
        if (!host_workspace.flat_seeds.empty()) {
            d_seed_records.upload_slice(
                host_workspace.flat_seeds.data(),
                host_workspace.flat_seeds.size());
        }
        const auto h2d_end = Clock::now();
        dynamic_h2d_seconds += elapsed_seconds(h2d_begin, h2d_end);
        dynamic_h2d_bytes +=
            checked_bytes(current_batch, sizeof(Vertex), "optimized G32 sources") +
            checked_bytes(batch_seed_count, sizeof(G32SeedRecord),
                          "optimized G32 sparse seeds");

        cuda_check(cudaEventRecord(event0), "cudaEventRecord G32 opt event0");

        g32_initialize_distance_kernel<<<initialize_blocks, generic_threads>>>(
            d_distance.data(),
            static_cast<std::uint64_t>(restricted_vertex_count));
        cuda_check(cudaGetLastError(), "optimized G32 initialize launch");
        cuda_check(cudaEventRecord(event1), "cudaEventRecord G32 opt event1");

        if (batch_seed_count != 0) {
            const unsigned scatter_blocks = checked_blocks(
                static_cast<std::uint64_t>(batch_seed_count),
                generic_threads,
                "optimized G32 seed scatter");
            g32_scatter_seed_kernel<<<scatter_blocks, generic_threads>>>(
                d_seed_records.data(),
                static_cast<std::uint64_t>(batch_seed_count),
                d_distance.data());
            cuda_check(cudaGetLastError(), "optimized G32 scatter launch");
        }
        cuda_check(cudaEventRecord(event2), "cudaEventRecord G32 opt event2");

        for (std::size_t level = 0;
             level + 1 < layout.level_offsets.size();
             ++level) {
            const Offset begin = layout.level_offsets[level];
            const Offset end = layout.level_offsets[level + 1];
            const std::uint64_t level_vertex_count =
                static_cast<std::uint64_t>(end - begin);
            if (level_vertex_count == 0) {
                continue;
            }
            const std::uint64_t thread_work =
                level_vertex_count * G32_SOURCE_LANES;
            const unsigned level_blocks = checked_blocks(
                thread_work, level_threads, "optimized G32 level pull");

            g32_level_pull_kernel<<<level_blocks, level_threads>>>(
                d_level_vertices.data() + static_cast<std::size_t>(begin),
                level_vertex_count,
                d_incoming_first.data(),
                d_incoming_tail.data(),
                d_incoming_weight.data(),
                static_cast<std::uint32_t>(current_batch),
                d_distance.data());
            cuda_check(cudaGetLastError(), "optimized G32 level pull launch");
            ++level_kernel_launches;
        }

        cuda_check(cudaEventRecord(event3), "cudaEventRecord G32 opt event3");

        g32_finalize_kernel<<<1, G32_SOURCE_LANES>>>(
            d_target_index.data(),
            d_target_ids.data(),
            static_cast<std::uint64_t>(targets.size()),
            d_batch_sources.data(),
            static_cast<std::uint32_t>(current_batch),
            d_distance.data(),
            quiet ? nullptr : d_batch_results.data(),
            d_checksums.data(),
            d_unreachable.data());
        cuda_check(cudaGetLastError(), "optimized G32 finalize launch");

        cuda_check(cudaEventRecord(event4), "cudaEventRecord G32 opt event4");
        cuda_check(cudaEventSynchronize(event4), "cudaEventSynchronize G32 opt");

        initialize_kernel_seconds += event_seconds(event0, event1);
        scatter_kernel_seconds += event_seconds(event1, event2);
        downward_kernel_seconds += event_seconds(event2, event3);
        finalize_kernel_seconds += event_seconds(event3, event4);
        device_sequence_seconds += event_seconds(event0, event4);

        const auto stats_begin = Clock::now();
        d_checksums.download_slice(batch_checksums.data(), current_batch);
        d_unreachable.download_slice(batch_unreachable.data(), current_batch);
        const auto stats_end = Clock::now();
        stats_d2h_seconds += elapsed_seconds(stats_begin, stats_end);
        stats_d2h_bytes +=
            checked_bytes(current_batch, sizeof(unsigned long long),
                          "optimized G32 checksums") +
            checked_bytes(current_batch, sizeof(unsigned long long),
                          "optimized G32 unreachable counts");

        const auto reduce_begin = Clock::now();
        for (std::size_t lane = 0; lane < current_batch; ++lane) {
            checksum ^= static_cast<std::uint64_t>(batch_checksums[lane]);
            unreachable += static_cast<std::uint64_t>(batch_unreachable[lane]);
        }
        const auto reduce_end = Clock::now();
        host_reduce_seconds += elapsed_seconds(reduce_begin, reduce_end);

        if (!quiet) {
            const std::size_t batch_result_count = current_batch * targets.size();
            const auto result_begin = Clock::now();
            d_batch_results.download_slice(
                host_results.data() + source_begin * targets.size(),
                batch_result_count);
            const auto result_end = Clock::now();
            result_d2h_seconds += elapsed_seconds(result_begin, result_end);
            result_d2h_bytes += checked_bytes(
                batch_result_count, sizeof(Weight), "optimized G32 results");
        }
    }

    const auto streaming_end = Clock::now();
    const double streaming_seconds = elapsed_seconds(streaming_begin, streaming_end);
    const auto query_wall_end = Clock::now();
    const double end_to_end_seconds =
        elapsed_seconds(query_wall_begin, query_wall_end);
    const std::uint64_t host_workspace_bytes_after =
        host_workspace.capacity_bytes();
    const std::uint64_t seed_buffer_peak_bytes =
        checked_bytes(d_seed_records.size(), sizeof(G32SeedRecord),
                      "optimized G32 peak seed buffer");
    const std::uint64_t device_query_peak_bytes =
        fixed_device_query_bytes + seed_buffer_peak_bytes;

    cudaEventDestroy(event0);
    cudaEventDestroy(event1);
    cudaEventDestroy(event2);
    cudaEventDestroy(event3);
    cudaEventDestroy(event4);

    const double all_h2d_seconds = restricted_h2d_seconds + dynamic_h2d_seconds;
    const std::uint64_t all_h2d_bytes = restricted_gpu_bytes + dynamic_h2d_bytes;
    const double all_d2h_seconds = stats_d2h_seconds + result_d2h_seconds;
    const std::uint64_t all_d2h_bytes = stats_d2h_bytes + result_d2h_bytes;

    const double average_device_ns =
        device_sequence_seconds * 1.0e9 / static_cast<double>(pair_count);
    const double average_end_to_end_ns =
        end_to_end_seconds * 1.0e9 / static_cast<double>(pair_count);

    if (print_report) {
        std::cerr
            << std::fixed << std::setprecision(6)
        << "G-RPHAST-32 OPTIMIZED timing breakdown\n"
        << "Reusable/static initialization (excluded)\n"
        << "  persistent host query workspace: "
        << reusable_host_workspace_init_seconds << " s\n"
        << "  CUDA runtime/device initialization: " << runtime_seconds << " s\n"
        << "Per-query CPU preprocessing\n"
        << "  timestamped target selection + selected-rank sort: "
        << selection_seconds << " s\n"
        << "  direct incoming-CSR + dependency-level build: "
        << layout_seconds << " s\n"
        << "  OpenMP upward CH searches (streamed batches): "
        << upward_seconds << " s\n"
        << "  sparse-seed flattening: " << seed_flatten_seconds << " s\n"
        << "Per-query GPU preparation\n"
        << "  target-sized CUDA allocations (COUNTED): "
        << allocation_seconds << " s\n"
        << "    seed-buffer growth subset: " << seed_buffer_growth_seconds << " s\n"
        << "  restricted graph H2D: " << restricted_h2d_seconds << " s ("
        << std::setprecision(2) << mib(restricted_gpu_bytes) << " MiB)\n"
        << std::setprecision(6)
        << "  source IDs + sparse seeds H2D: " << dynamic_h2d_seconds << " s ("
        << std::setprecision(2) << mib(dynamic_h2d_bytes) << " MiB)\n"
        << std::setprecision(6)
        << "CUDA device stages\n"
        << "  distance initialization kernels: " << initialize_kernel_seconds << " s\n"
        << "  sparse-seed scatter kernels: " << scatter_kernel_seconds << " s\n"
        << "  level-by-level downward pull: " << downward_kernel_seconds << " s\n"
        << "  finalization/checksum kernels: " << finalize_kernel_seconds << " s\n"
        << "  complete CUDA device sequence: " << device_sequence_seconds << " s\n"
        << "  level kernel launches: " << level_kernel_launches << "\n"
        << "Device -> host / validation\n"
        << "  checksum/unreachable D2H: " << stats_d2h_seconds << " s ("
        << std::setprecision(2) << mib(stats_d2h_bytes) << " MiB)\n"
        << std::setprecision(6)
        << "  distance-result D2H: " << result_d2h_seconds << " s ("
        << std::setprecision(2) << mib(result_d2h_bytes) << " MiB)"
        << (quiet ? " [quiet: disabled]" : "") << "\n"
        << std::setprecision(6)
        << "  host checksum reduction: " << host_reduce_seconds << " s\n"
        << "Streaming source-batch phase wall time: " << streaming_seconds << " s\n"
        << "Transfers summary\n"
        << "  all H2D: " << all_h2d_seconds << " s ("
        << std::setprecision(2) << mib(all_h2d_bytes) << " MiB)\n"
        << std::setprecision(6)
        << "  all D2H: " << all_d2h_seconds << " s ("
        << std::setprecision(2) << mib(all_d2h_bytes) << " MiB)\n"
        << std::setprecision(6)
        << "Benchmark aggregate (CH construction/file I/O excluded)\n"
        << "  per-query END-TO-END wall time: " << end_to_end_seconds << " s\n"
        << "    = selection + direct layout + target-sized allocation/upload"
           " + streamed CPU upward/GPU query batches\n"
        << "  pure CUDA device sequence: " << device_sequence_seconds << " s\n"
        << "  CUDA batches executed: " << batch_count << "\n"
        << "  sparse upward seeds: " << total_seed_count << " total\n"
        << "  restricted graph: " << restricted_vertex_count << " vertices, "
        << layout.incoming_tail.size() << " arcs, "
        << layout.level_offsets.size() - 1 << " levels\n"
        << std::setprecision(2)
        << "  average pure device time: " << average_device_ns << " ns/pair\n"
        << "  average END-TO-END query time: "
        << average_end_to_end_ns << " ns/pair\n"
        << "pairs completed: " << pair_count
        << ", unreachable: " << unreachable
        << ", distance checksum: " << checksum << "\n";
    }

    G32OptimizedQueryStats stats;
    stats.source_count = static_cast<std::uint64_t>(sources.size());
    stats.target_count = static_cast<std::uint64_t>(targets.size());
    stats.pair_count = pair_count;
    stats.runtime_seconds = runtime_seconds;
    stats.selection_seconds = selection_seconds;
    stats.layout_seconds = layout_seconds;
    stats.allocation_seconds = allocation_seconds;
    stats.seed_buffer_growth_seconds = seed_buffer_growth_seconds;
    stats.restricted_h2d_seconds = restricted_h2d_seconds;
    stats.upward_seconds = upward_seconds;
    stats.seed_flatten_seconds = seed_flatten_seconds;
    stats.dynamic_h2d_seconds = dynamic_h2d_seconds;
    stats.initialize_kernel_seconds = initialize_kernel_seconds;
    stats.scatter_kernel_seconds = scatter_kernel_seconds;
    stats.downward_kernel_seconds = downward_kernel_seconds;
    stats.finalize_kernel_seconds = finalize_kernel_seconds;
    stats.device_sequence_seconds = device_sequence_seconds;
    stats.stats_d2h_seconds = stats_d2h_seconds;
    stats.result_d2h_seconds = result_d2h_seconds;
    stats.host_reduce_seconds = host_reduce_seconds;
    stats.streaming_seconds = streaming_seconds;
    stats.end_to_end_seconds = end_to_end_seconds;
    stats.restricted_h2d_bytes = restricted_gpu_bytes;
    stats.dynamic_h2d_bytes = dynamic_h2d_bytes;
    stats.stats_d2h_bytes = stats_d2h_bytes;
    stats.result_d2h_bytes = result_d2h_bytes;
    stats.total_seed_count = total_seed_count;
    stats.checksum = checksum;
    stats.unreachable = unreachable;
    stats.batch_count = batch_count;
    stats.level_kernel_launches = level_kernel_launches;
    stats.restricted_vertex_count = static_cast<std::uint64_t>(restricted_vertex_count);
    stats.restricted_arc_count = static_cast<std::uint64_t>(layout.incoming_tail.size());
    stats.restricted_level_count = static_cast<std::uint64_t>(layout.level_offsets.size() - 1);
    stats.used_full_rank_scan = host_workspace.selection.used_full_rank_scan();
    stats.distance_workspace_bytes = distance_workspace_bytes;
    stats.seed_buffer_peak_bytes = seed_buffer_peak_bytes;
    stats.device_query_peak_bytes = device_query_peak_bytes;
    stats.device_free_before_bytes =
        static_cast<std::uint64_t>(device_free_before);
    stats.host_layout_bytes = host_layout_bytes;
    stats.host_workspace_bytes_before = host_workspace_bytes_before;
    stats.host_workspace_bytes_after = host_workspace_bytes_after;
    stats.host_auxiliary_bytes = host_auxiliary_bytes;

    if (!quiet) {
        for (std::size_t source_index = 0;
             source_index < sources.size(); ++source_index) {
            for (std::size_t target_index = 0;
                 target_index < targets.size(); ++target_index) {
                print_distance_result(
                    sources[source_index],
                    targets[target_index],
                    host_results[source_index * targets.size() + target_index]);
            }
        }
    }

    return stats;
}

static void write_g32_benchmark_csv_header(std::ostream& out) {
    out
        << "format_version,input_file,device,base_seed,run_index,query_seed,kernel,kernel_order,"
        << "source_count,target_count,pair_count,"
        << "resident_hl_bytes,persistent_workspace_bytes,selected_label_estimated_bytes,"
        << "host_query_structure_peak_bytes,host_total_estimated_peak_bytes,"
        << "device_required_bytes,device_free_before_bytes,"
        << "target_bucket_count,common_hubs,source_work_items,candidate_relaxations,maximum_target_fanout,"
        << "resident_selection_s,preprocess_s,h2d_allocation_s,gpu_init_s,gpu_relax_s,gpu_finalize_s,"
        << "gpu_total_s,d2h_s,total_post_resident_s,total_ns_per_pair,total_pairs_per_s,"
        << "candidate_relaxations_per_s,unreachable,checksum,"
        << "algorithm_family,threads,gpu_batch_sources,vertex_count,query_generation_s,"
        << "ch_read_s_excluded,persistent_init_s_excluded,gpu_total_global_mem_bytes,"
        << "selection_s,layout_s,cuda_allocation_s,seed_buffer_growth_s,restricted_h2d_s,"
        << "upward_s,seed_flatten_s,dynamic_h2d_s,initialize_kernel_s,scatter_kernel_s,"
        << "downward_kernel_s,finalize_kernel_s,device_sequence_s,stats_d2h_s,result_d2h_s,"
        << "host_reduce_s,streaming_s,device_ns_per_pair,"
        << "restricted_vertices,restricted_arcs,restricted_levels,used_full_rank_scan,"
        << "total_seeds,cuda_batches,level_kernel_launches,ch_host_logical_bytes,"
        << "persistent_host_workspace_before_bytes,persistent_host_workspace_after_bytes,"
        << "query_host_layout_bytes,query_host_auxiliary_bytes,distance_workspace_bytes,"
        << "seed_buffer_peak_bytes,restricted_h2d_bytes,dynamic_h2d_bytes,stats_d2h_bytes,"
        << "result_d2h_bytes,ch_forward_settled,ch_backward_settled,ch_relaxed_arcs,"
        << "ch_pair_workspace_bytes\n";
}

static std::uint64_t benchmark_query_seed(
    std::uint64_t base_seed,
    std::uint64_t index,
    bool warmup) {

    const std::uint64_t domain = warmup
        ? 0x6a09e667f3bcc909ULL
        : 0xbb67ae8584caa73bULL;
    return mix_query_value(
        base_seed ^ domain ^ mix_query_value(index + 0x9e3779b97f4a7c15ULL));
}

static void write_g32_benchmark_csv_row(
    std::ostream& out,
    std::uint64_t run_index,
    std::uint64_t base_seed,
    std::uint64_t query_seed,
    const std::string& ch_path,
    const std::string& gpu_name,
    int device_id,
    int requested_threads,
    std::size_t batch_sources,
    std::uint64_t vertex_count,
    double generation_seconds,
    double ch_read_seconds,
    double persistent_init_seconds,
    std::uint64_t gpu_total_global_mem_bytes,
    std::uint64_t ch_bytes,
    const G32OptimizedQueryStats& stats) {

    const double total_ns_per_pair =
        stats.pair_count == 0
            ? 0.0
            : stats.end_to_end_seconds * 1.0e9 /
                  static_cast<double>(stats.pair_count);
    const double total_pairs_per_s =
        stats.end_to_end_seconds <= 0.0
            ? 0.0
            : static_cast<double>(stats.pair_count) / stats.end_to_end_seconds;
    const double device_ns_per_pair =
        stats.pair_count == 0
            ? 0.0
            : stats.device_sequence_seconds * 1.0e9 /
                  static_cast<double>(stats.pair_count);

    const std::uint64_t persistent_workspace_bytes =
        stats.host_workspace_bytes_after;
    const std::uint64_t host_query_structure_peak_bytes =
        stats.host_layout_bytes + stats.host_auxiliary_bytes;
    const std::uint64_t host_total_estimated_peak_bytes =
        ch_bytes + persistent_workspace_bytes + host_query_structure_peak_bytes;

    const double resident_selection_s = stats.selection_seconds;
    const double preprocess_s =
        stats.layout_seconds + stats.upward_seconds + stats.seed_flatten_seconds;
    const double h2d_allocation_s =
        stats.allocation_seconds + stats.restricted_h2d_seconds +
        stats.dynamic_h2d_seconds;
    const double gpu_init_s =
        stats.initialize_kernel_seconds + stats.scatter_kernel_seconds;
    const double gpu_relax_s = stats.downward_kernel_seconds;
    const double gpu_finalize_s = stats.finalize_kernel_seconds;
    const double gpu_total_s = stats.device_sequence_seconds;
    const double d2h_s = stats.stats_d2h_seconds + stats.result_d2h_seconds;

    out << std::setprecision(17)
        << 1 << ','
        << csv_escape(ch_path) << ','
        << csv_escape(gpu_name) << ','
        << base_seed << ','
        << run_index << ','
        << query_seed << ','
        << "cuda-g32-opt" << ','
        << 0 << ','
        << stats.source_count << ','
        << stats.target_count << ','
        << stats.pair_count << ','
        << 0 << ','
        << persistent_workspace_bytes << ','
        << 0 << ','
        << host_query_structure_peak_bytes << ','
        << host_total_estimated_peak_bytes << ','
        << stats.device_query_peak_bytes << ','
        << stats.device_free_before_bytes << ','
        << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ','
        << resident_selection_s << ','
        << preprocess_s << ','
        << h2d_allocation_s << ','
        << gpu_init_s << ','
        << gpu_relax_s << ','
        << gpu_finalize_s << ','
        << gpu_total_s << ','
        << d2h_s << ','
        << stats.end_to_end_seconds << ','
        << total_ns_per_pair << ','
        << total_pairs_per_s << ','
        << 0.0 << ','
        << stats.unreachable << ','
        << stats.checksum << ','
        << "rphast" << ','
        << requested_threads << ','
        << batch_sources << ','
        << vertex_count << ','
        << generation_seconds << ','
        << ch_read_seconds << ','
        << persistent_init_seconds << ','
        << gpu_total_global_mem_bytes << ','
        << stats.selection_seconds << ','
        << stats.layout_seconds << ','
        << stats.allocation_seconds << ','
        << stats.seed_buffer_growth_seconds << ','
        << stats.restricted_h2d_seconds << ','
        << stats.upward_seconds << ','
        << stats.seed_flatten_seconds << ','
        << stats.dynamic_h2d_seconds << ','
        << stats.initialize_kernel_seconds << ','
        << stats.scatter_kernel_seconds << ','
        << stats.downward_kernel_seconds << ','
        << stats.finalize_kernel_seconds << ','
        << stats.device_sequence_seconds << ','
        << stats.stats_d2h_seconds << ','
        << stats.result_d2h_seconds << ','
        << stats.host_reduce_seconds << ','
        << stats.streaming_seconds << ','
        << device_ns_per_pair << ','
        << stats.restricted_vertex_count << ','
        << stats.restricted_arc_count << ','
        << stats.restricted_level_count << ','
        << (stats.used_full_rank_scan ? 1 : 0) << ','
        << stats.total_seed_count << ','
        << stats.batch_count << ','
        << stats.level_kernel_launches << ','
        << ch_bytes << ','
        << stats.host_workspace_bytes_before << ','
        << stats.host_workspace_bytes_after << ','
        << stats.host_layout_bytes << ','
        << stats.host_auxiliary_bytes << ','
        << stats.distance_workspace_bytes << ','
        << stats.seed_buffer_peak_bytes << ','
        << stats.restricted_h2d_bytes << ','
        << stats.dynamic_h2d_bytes << ','
        << stats.stats_d2h_bytes << ','
        << stats.result_d2h_bytes << ','
        << 0 << ',' << 0 << ',' << 0 << ',' << 0 << '\n';
}

static void write_ch_benchmark_csv_row(
    std::ostream& out,
    std::uint64_t run_index,
    std::uint64_t base_seed,
    std::uint64_t query_seed,
    const std::string& ch_path,
    int requested_threads,
    std::uint64_t vertex_count,
    double generation_seconds,
    double ch_read_seconds,
    double persistent_init_seconds,
    std::uint64_t ch_bytes,
    const ParallelCHQueryStats& stats) {

    const double total_ns_per_pair =
        stats.pair_count == 0
            ? 0.0
            : stats.end_to_end_seconds * 1.0e9 /
                  static_cast<double>(stats.pair_count);
    const double total_pairs_per_s =
        stats.end_to_end_seconds <= 0.0
            ? 0.0
            : static_cast<double>(stats.pair_count) / stats.end_to_end_seconds;
    const std::uint64_t persistent_workspace_bytes = stats.workspace_bytes_after;
    const std::uint64_t host_query_structure_peak_bytes = 0;
    const std::uint64_t host_total_estimated_peak_bytes =
        ch_bytes + persistent_workspace_bytes;

    out << std::setprecision(17)
        // HL-compatible common schema.
        << 1 << ','
        << csv_escape(ch_path) << ','
        << "CPU/OpenMP" << ','
        << base_seed << ','
        << run_index << ','
        << query_seed << ','
        << "ch-parallel" << ','
        << 0 << ','
        << stats.source_count << ','
        << stats.target_count << ','
        << stats.pair_count << ','
        << 0 << ','
        << persistent_workspace_bytes << ','
        << 0 << ','
        << host_query_structure_peak_bytes << ','
        << host_total_estimated_peak_bytes << ','
        << 0 << ','
        << 0 << ','
        << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ','
        << 0.0 << ',' // resident_selection_s
        << 0.0 << ',' // preprocess_s
        << 0.0 << ',' // h2d_allocation_s
        << 0.0 << ',' // gpu_init_s
        << 0.0 << ',' // gpu_relax_s
        << 0.0 << ',' // gpu_finalize_s
        << 0.0 << ',' // gpu_total_s
        << 0.0 << ',' // d2h_s
        << stats.end_to_end_seconds << ','
        << total_ns_per_pair << ','
        << total_pairs_per_s << ','
        << 0.0 << ','
        << stats.unreachable << ','
        << stats.checksum << ','
        // CH/RPHAST-specific diagnostics.
        << "ch" << ','
        << requested_threads << ','
        << 0 << ','
        << vertex_count << ','
        << generation_seconds << ','
        << ch_read_seconds << ','
        << persistent_init_seconds << ','
        << 0 << ','
        // Native G32 timing columns.
        << 0.0 << ',' << 0.0 << ',' << 0.0 << ',' << 0.0 << ','
        << 0.0 << ',' << 0.0 << ',' << 0.0 << ',' << 0.0 << ','
        << 0.0 << ',' << 0.0 << ',' << 0.0 << ',' << 0.0 << ','
        << 0.0 << ',' << 0.0 << ',' << 0.0 << ',' << 0.0 << ','
        << 0.0 << ',' << 0.0 << ','
        << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ','
        << 0 << ',' << 0 << ',' << 0 << ','
        << ch_bytes << ','
        << stats.workspace_bytes_before << ','
        << stats.workspace_bytes_after << ','
        << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ','
        << 0 << ',' << 0 << ',' << 0 << ',' << 0 << ','
        << stats.forward_settled << ','
        << stats.backward_settled << ','
        << stats.relaxed_arcs << ','
        << stats.workspace_bytes_after << '\n';
}

static int run_g32_random_benchmark_cli(int argc, char** argv) {
    using Clock = std::chrono::steady_clock;

    if (argc < 3) {
        throw std::runtime_error(
            "benchmark requires an input CH file and benchmark options");
    }

    const std::string ch_path = argv[2];
#ifdef _OPENMP
    int requested_threads = omp_get_max_threads();
#else
    int requested_threads = 1;
#endif
    int device_id = 0;
    std::size_t gpu_batch_sources = G32_SOURCE_LANES;
    std::string benchmark_backend = "cuda-g32-opt";
    std::uint64_t query_count = 0;
    std::uint64_t warmup_count = 0;
    std::uint64_t source_count = 0;
    std::uint64_t target_count = 0;
    std::uint64_t master_seed = 0;
    bool seed_seen = false;
    std::string output_path;

    for (int i = 3; i < argc; ++i) {
        const std::string option = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string(name) + " requires a value");
            }
            return argv[++i];
        };
        auto parse_u64 = [](const std::string& value, const char* name) {
            std::size_t parsed = 0;
            const std::uint64_t result = std::stoull(value, &parsed);
            if (parsed != value.size()) {
                throw std::runtime_error(std::string(name) + " must be an unsigned integer");
            }
            return result;
        };

        if (option == "--queries") {
            query_count = parse_u64(require_value("--queries"), "--queries");
        } else if (option == "--warmup") {
            warmup_count = parse_u64(require_value("--warmup"), "--warmup");
        } else if (option == "--sources") {
            source_count = parse_u64(require_value("--sources"), "--sources");
        } else if (option == "--targets") {
            target_count = parse_u64(require_value("--targets"), "--targets");
        } else if (option == "--output") {
            output_path = require_value("--output");
            if (output_path.empty()) {
                throw std::runtime_error("--output must not be empty");
            }
        } else if (option == "--seed") {
            master_seed = parse_u64(require_value("--seed"), "--seed");
            seed_seen = true;
        } else if (option == "--threads") {
            const std::uint64_t value =
                parse_u64(require_value("--threads"), "--threads");
            if (value == 0 || value > static_cast<std::uint64_t>(
                    std::numeric_limits<int>::max())) {
                throw std::runtime_error("--threads must be a positive integer fitting in int");
            }
            requested_threads = static_cast<int>(value);
        } else if (option == "--device") {
            const std::uint64_t value =
                parse_u64(require_value("--device"), "--device");
            if (value > static_cast<std::uint64_t>(
                    std::numeric_limits<int>::max())) {
                throw std::runtime_error("--device must fit in int");
            }
            device_id = static_cast<int>(value);
        } else if (option == "--gpu-batch-sources") {
            const std::uint64_t value = parse_u64(
                require_value("--gpu-batch-sources"), "--gpu-batch-sources");
            if (value == 0 || value > G32_SOURCE_LANES) {
                throw std::runtime_error(
                    "--gpu-batch-sources must be in [1,32] for cuda-g32-opt");
            }
            gpu_batch_sources = static_cast<std::size_t>(value);
        } else if (option == "--backend") {
            const std::string value = require_value("--backend");
            if (value != "cuda-g32-opt" && value != "ch-parallel") {
                throw std::runtime_error(
                    "benchmark --backend must be cuda-g32-opt or ch-parallel");
            }
            benchmark_backend = value;
        } else {
            throw std::runtime_error("Unknown benchmark option: " + option);
        }
    }

    if (query_count == 0) {
        throw std::runtime_error("--queries must be a positive integer");
    }
    if (source_count == 0 || target_count == 0) {
        throw std::runtime_error("--sources and --targets must both be positive");
    }
    if (output_path.empty()) {
        throw std::runtime_error("benchmark requires --output <results.csv>");
    }

    if (!seed_seen) {
        std::random_device random_device;
        master_seed =
            (static_cast<std::uint64_t>(random_device()) << 32) ^
            static_cast<std::uint64_t>(random_device());
    }

    const auto ch_read_begin = Clock::now();
    CHQueryGraph ch = read_ch_text(ch_path);
    const auto ch_read_end = Clock::now();
    const double ch_read_seconds =
        std::chrono::duration<double>(ch_read_end - ch_read_begin).count();
    const std::size_t n = ch.upward.num_vertices();
    if (source_count > n || target_count > n) {
        throw std::runtime_error(
            "--sources/--targets cannot exceed the number of graph vertices");
    }

#ifdef _OPENMP
    omp_set_dynamic(0);
#pragma omp parallel num_threads(requested_threads)
    {
        // Warm the OpenMP runtime before any measured query.
    }
#endif

    const auto reusable_begin = Clock::now();
    std::unique_ptr<G32OptimizedHostWorkspace> host_workspace;
    std::unique_ptr<ParallelCHWorkspace> ch_workspace;
    cudaDeviceProp properties{};
    std::string gpu_name;
    std::uint64_t gpu_total_global_mem_bytes = 0;

    if (benchmark_backend == "cuda-g32-opt") {
        host_workspace =
            std::make_unique<G32OptimizedHostWorkspace>(n, requested_threads);
        cuda_check(cudaSetDevice(device_id), "cudaSetDevice benchmark");
        cuda_check(cudaFree(nullptr), "CUDA runtime warmup benchmark");
        cuda_check(
            cudaGetDeviceProperties(&properties, device_id),
            "cudaGetDeviceProperties benchmark");
        gpu_name = properties.name;
        gpu_total_global_mem_bytes =
            static_cast<std::uint64_t>(properties.totalGlobalMem);
    } else {
        ch_workspace =
            std::make_unique<ParallelCHWorkspace>(n, requested_threads);
    }

    const auto reusable_end = Clock::now();
    const double reusable_init_seconds =
        std::chrono::duration<double>(reusable_end - reusable_begin).count();

    std::ofstream output(output_path, std::ios::out | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Cannot open benchmark output file: " + output_path);
    }
    write_g32_benchmark_csv_header(output);

    const std::uint64_t ch_bytes = ch_logical_storage_bytes(ch);
    const std::uint64_t total_runs = warmup_count + query_count;

    std::cerr
        << std::fixed << std::setprecision(6)
        << "Random many-to-many benchmark\n"
        << "  backend: " << benchmark_backend << "\n"
        << "  graph: " << ch_path << "\n"
        << "  CH read/parse (excluded): " << ch_read_seconds << " s\n"
        << "  persistent workspace/runtime init (excluded): "
        << reusable_init_seconds << " s\n";

    if (benchmark_backend == "cuda-g32-opt") {
        std::cerr << "  GPU: " << gpu_name << "\n";
    } else {
        std::cerr << "  CH pair workers: " << requested_threads << "\n";
    }

    std::cerr
        << "  queries: " << query_count << ", warmups: " << warmup_count << "\n"
        << "  |S|=" << source_count << ", |T|=" << target_count << "\n"
        << "  base seed: " << master_seed << "\n"
        << "  CSV: " << output_path << "\n";

    std::vector<Vertex> sources;
    std::vector<Vertex> targets;

    for (std::uint64_t execution_index = 0;
         execution_index < total_runs;
         ++execution_index) {
        const bool warmup = execution_index < warmup_count;
        const std::uint64_t logical_run_index =
            warmup ? execution_index : execution_index - warmup_count;
        const std::uint64_t query_seed = benchmark_query_seed(
            master_seed, logical_run_index, warmup);
        std::mt19937_64 query_generator(query_seed);

        const auto generation_begin = Clock::now();
        sources = generate_random_vertices(
            static_cast<std::size_t>(source_count), n, query_generator);
        targets = generate_random_vertices(
            static_cast<std::size_t>(target_count), n, query_generator);
        const auto generation_end = Clock::now();
        const double generation_seconds =
            std::chrono::duration<double>(generation_end - generation_begin).count();

        G32OptimizedQueryStats g32_stats;
        ParallelCHQueryStats ch_stats;

        if (benchmark_backend == "cuda-g32-opt") {
            if (!host_workspace) {
                throw std::runtime_error("Missing G32 benchmark workspace");
            }
            g32_stats = run_many_to_many_cuda_g32_optimized(
                ch,
                *host_workspace,
                sources,
                targets,
                true,
                device_id,
                gpu_batch_sources,
                requested_threads,
                reusable_init_seconds,
                false);
        } else {
            if (!ch_workspace) {
                throw std::runtime_error("Missing parallel CH benchmark workspace");
            }
            ch_stats = run_many_to_many_parallel_ch(
                ch,
                *ch_workspace,
                sources,
                targets,
                true,
                requested_threads,
                false);
        }

        const double measured_seconds =
            benchmark_backend == "cuda-g32-opt"
                ? g32_stats.end_to_end_seconds
                : ch_stats.end_to_end_seconds;

        if (warmup) {
            std::cerr
                << "  warmup " << (execution_index + 1) << '/' << warmup_count
                << ": " << measured_seconds * 1000.0 << " ms\n";
            continue;
        }

        const std::uint64_t query_index = logical_run_index;
        if (benchmark_backend == "cuda-g32-opt") {
            write_g32_benchmark_csv_row(
                output,
                query_index,
                master_seed,
                query_seed,
                ch_path,
                gpu_name,
                device_id,
                requested_threads,
                gpu_batch_sources,
                static_cast<std::uint64_t>(n),
                generation_seconds,
                ch_read_seconds,
                reusable_init_seconds,
                gpu_total_global_mem_bytes,
                ch_bytes,
                g32_stats);
        } else {
            write_ch_benchmark_csv_row(
                output,
                query_index,
                master_seed,
                query_seed,
                ch_path,
                requested_threads,
                static_cast<std::uint64_t>(n),
                generation_seconds,
                ch_read_seconds,
                reusable_init_seconds,
                ch_bytes,
                ch_stats);
        }
        output.flush();

        std::cerr
            << "  query " << (query_index + 1) << '/' << query_count
            << ": " << measured_seconds * 1000.0 << " ms";
        if (benchmark_backend == "cuda-g32-opt") {
            std::cerr
                << ", restricted=" << g32_stats.restricted_vertex_count << " V / "
                << g32_stats.restricted_arc_count << " A"
                << ", device_peak="
                << std::setprecision(2)
                << (static_cast<double>(g32_stats.device_query_peak_bytes) /
                    (1024.0 * 1024.0))
                << " MiB";
        } else {
            std::cerr
                << ", settled="
                << (ch_stats.forward_settled + ch_stats.backward_settled)
                << ", workspace=" << std::setprecision(2)
                << (static_cast<double>(ch_stats.workspace_bytes_after) /
                    (1024.0 * 1024.0))
                << " MiB";
        }
        std::cerr << "\n" << std::setprecision(6);
    }

    std::cerr << "Benchmark complete: " << output_path << "\n";
    return 0;
}

#endif // __CUDACC__

static void print_usage(const char* program) {
    std::cerr
        << "Usage:\n"
        << "  " << program
        << " benchmark <input.ch.txt> --queries Q --sources N --targets M --output results.csv"
           " [--warmup W] [--seed S] [--threads N] [--device N]"
           " [--gpu-batch-sources N] [--backend cuda-g32-opt|ch-parallel]\n"
        << "  " << program
        << " query <input.ch.txt> <sources.txt> <targets.txt>"
           " [--backend cpu|cpu-rank|ch-parallel|cuda|cuda-g32|cuda-g32-opt] [--threads N] [--device N]"
           " [--gpu-batch-sources N] [--quiet]\n"
        << "  " << program
        << " query <input.ch.txt> --random N M"
           " [--seed S] [--backend cpu|cpu-rank|ch-parallel|cuda|cuda-g32|cuda-g32-opt] [--threads N]"
           " [--device N] [--gpu-batch-sources N] [--quiet]\n\n"
        << "Benchmark mode:\n"
        << "  Loads the CH once, then runs Q independent random many-to-many queries.\n"
        << "  Random S/T generation is outside the per-query timer but is recorded in CSV.\n"
        << "  One CSV row is written per measured query; --warmup runs are not written.\n"
        << "  Measured query_seed values depend only on --seed and run_index; warmup count does not shift them.\n"
        << "  The first CSV columns match the HL benchmark schema (format_version=1).\n"
        << "  CH/RPHAST-specific diagnostics are appended after the common columns.\n"
        << "  Benchmark backends: cuda-g32-opt and ch-parallel.\n\n"
        << "Input and representation match the HL implementation:\n"
        << "  CH format: CH_TEXT_V2\n"
        << "  source/target files: whitespace-separated 1-based vertex IDs\n"
        << "  Vertex: uint32_t, Weight: uint64_t, Offset: uint64_t\n"
        << "  INF: max(uint64_t)/4\n\n"
        << "Backends:\n"
        << "  cpu      canonical RPHAST query: upward CH Dijkstra + restricted sweep\n"
        << "  cpu-rank exact rank-sweep reference matching the CUDA kernel logic\n"
        << "  ch-parallel regular bidirectional CH query per s-t pair, parallelized with OpenMP\n"
        << "  cuda     naive CUDA baseline: one thread/source, dense upward rank sweep\n"
        << "  cuda-g32 G-RPHAST-32: OpenMP upward CH searches + warp/vertex GPU pull\n"
        << "  cuda-g32-opt optimized G-RPHAST-32: timestamped selection + direct layout + streaming batches\n"
        << "           --gpu-batch-sources defaults to 32 and may be set to 1..32\n\n"
        << "Non-quiet output keeps four columns for compatibility with the HL tool:\n"
        << "  source target distance 0\n"
        << "The fourth column is always zero because RPHAST has no meeting hub.\n";
}

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    try {
        if (argc < 2) {
            print_usage(argv[0]);
            return 1;
        }

        const std::string command = argv[1];
        if (command == "benchmark") {
#ifdef __CUDACC__
            return run_g32_random_benchmark_cli(argc, argv);
#else
            throw std::runtime_error(
                "benchmark mode requires a CUDA build with nvcc");
#endif
        }

        if (command != "query") {
            print_usage(argv[0]);
            return 1;
        }

        if (argc < 4) {
            print_usage(argv[0]);
            return 1;
        }

        const std::string ch_path = argv[2];
        std::string backend = "cpu";
#ifdef _OPENMP
        int requested_threads = omp_get_max_threads();
#else
        int requested_threads = 1;
#endif
        int device_id = 0;
        std::size_t gpu_batch_sources = 0;
        bool quiet = false;
        bool random_mode = false;
        bool seed_seen = false;
        std::uint64_t seed = 0;
        std::uint64_t random_source_count = 0;
        std::uint64_t random_target_count = 0;
        std::vector<std::string> positional;

        for (int i = 3; i < argc; ++i) {
            const std::string option = argv[i];

            if (option == "--backend") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--backend requires a value");
                }
                backend = argv[++i];
                if (backend != "cpu" &&
                    backend != "cpu-rank" &&
                    backend != "ch-parallel" &&
                    backend != "cuda" &&
                    backend != "cuda-g32" &&
                    backend != "cuda-g32-opt") {
                    throw std::runtime_error(
                        "--backend must be cpu, cpu-rank, ch-parallel, cuda, cuda-g32, or cuda-g32-opt");
                }
                continue;
            }

            if (option == "--threads") {
                if (i + 1 >= argc) {
                    throw std::runtime_error(
                        "--threads requires a positive integer");
                }
                const std::string value = argv[++i];
                std::size_t parsed = 0;
                const std::uint64_t parsed_threads =
                    std::stoull(value, &parsed);
                if (parsed != value.size() ||
                    parsed_threads == 0 ||
                    parsed_threads >
                        static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max())) {
                    throw std::runtime_error(
                        "--threads must be a positive integer that fits in int");
                }
                requested_threads = static_cast<int>(parsed_threads);
                continue;
            }

            if (option == "--device") {
                if (i + 1 >= argc) {
                    throw std::runtime_error(
                        "--device requires a non-negative integer");
                }
                const std::string value = argv[++i];
                std::size_t parsed = 0;
                const std::uint64_t parsed_device =
                    std::stoull(value, &parsed);
                if (parsed != value.size() ||
                    parsed_device >
                        static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max())) {
                    throw std::runtime_error(
                        "--device must fit in int");
                }
                device_id = static_cast<int>(parsed_device);
                continue;
            }

            if (option == "--gpu-batch-sources") {
                if (i + 1 >= argc) {
                    throw std::runtime_error(
                        "--gpu-batch-sources requires a positive integer");
                }
                const std::string value = argv[++i];
                std::size_t parsed = 0;
                const std::uint64_t parsed_batch =
                    std::stoull(value, &parsed);
                if (parsed != value.size() || parsed_batch == 0 ||
                    parsed_batch >
                        static_cast<std::uint64_t>(
                            std::numeric_limits<std::size_t>::max())) {
                    throw std::runtime_error(
                        "--gpu-batch-sources must be a positive size_t value");
                }
                gpu_batch_sources = static_cast<std::size_t>(parsed_batch);
                continue;
            }

            if (option == "--quiet") {
                quiet = true;
                continue;
            }

            if (option == "--seed") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--seed requires an integer");
                }
                const std::string value = argv[++i];
                std::size_t parsed = 0;
                seed = std::stoull(value, &parsed);
                if (parsed != value.size()) {
                    throw std::runtime_error(
                        "--seed must be an unsigned integer");
                }
                seed_seen = true;
                continue;
            }

            if (option == "--random") {
                if (i + 2 >= argc) {
                    throw std::runtime_error("--random requires N and M");
                }
                const std::string n_value = argv[++i];
                const std::string m_value = argv[++i];
                std::size_t n_parsed = 0;
                std::size_t m_parsed = 0;
                random_source_count = std::stoull(n_value, &n_parsed);
                random_target_count = std::stoull(m_value, &m_parsed);
                if (n_parsed != n_value.size() ||
                    m_parsed != m_value.size() ||
                    random_source_count == 0 ||
                    random_target_count == 0) {
                    throw std::runtime_error(
                        "--random N M requires two positive integers");
                }
                random_mode = true;
                continue;
            }

            if (option.rfind("--", 0) == 0) {
                throw std::runtime_error(
                    "Unknown query option: " + option);
            }

            positional.push_back(option);
        }

#ifndef __CUDACC__
        if (backend == "cuda" || backend == "cuda-g32" || backend == "cuda-g32-opt") {
            throw std::runtime_error(
                "CUDA backend requested, but this executable was not built with nvcc");
        }
#endif

        const auto ch_read_begin = std::chrono::steady_clock::now();
        CHQueryGraph ch = read_ch_text(ch_path);
        const auto ch_read_end = std::chrono::steady_clock::now();
        const double ch_read_seconds =
            std::chrono::duration<double>(
                ch_read_end - ch_read_begin).count();

        const std::size_t n = ch.upward.num_vertices();

        std::unique_ptr<G32OptimizedHostWorkspace> g32_opt_host_workspace;
        std::unique_ptr<ParallelCHWorkspace> parallel_ch_workspace;
        double g32_opt_reusable_init_seconds = 0.0;
        double parallel_ch_reusable_init_seconds = 0.0;
        if (backend == "cuda-g32-opt") {
            const auto reusable_begin = std::chrono::steady_clock::now();
#ifdef _OPENMP
            omp_set_dynamic(0);
#pragma omp parallel num_threads(requested_threads)
            {
                // Force OpenMP worker/runtime creation before query timing.
            }
#endif
            g32_opt_host_workspace =
                std::make_unique<G32OptimizedHostWorkspace>(n, requested_threads);
#ifdef __CUDACC__
            cuda_check(cudaSetDevice(device_id), "cudaSetDevice reusable G32 opt");
            cuda_check(cudaFree(nullptr), "CUDA runtime warmup reusable G32 opt");
#endif
            const auto reusable_end = std::chrono::steady_clock::now();
            g32_opt_reusable_init_seconds =
                std::chrono::duration<double>(reusable_end - reusable_begin).count();
        } else if (backend == "ch-parallel") {
            const auto reusable_begin = std::chrono::steady_clock::now();
#ifdef _OPENMP
            omp_set_dynamic(0);
#pragma omp parallel num_threads(requested_threads)
            {
                // Force OpenMP worker/runtime creation before query timing.
            }
#endif
            parallel_ch_workspace =
                std::make_unique<ParallelCHWorkspace>(n, requested_threads);
            const auto reusable_end = std::chrono::steady_clock::now();
            parallel_ch_reusable_init_seconds =
                std::chrono::duration<double>(reusable_end - reusable_begin).count();
        }

        std::vector<Vertex> sources;
        std::vector<Vertex> targets;

        const auto query_input_begin = std::chrono::steady_clock::now();

        if (random_mode) {
            if (!positional.empty()) {
                throw std::runtime_error(
                    "Do not combine --random with source/target files");
            }
            if (random_source_count > n || random_target_count > n) {
                throw std::runtime_error(
                    "--random cannot request more unique vertices than the graph contains");
            }

            if (!seed_seen) {
                std::random_device random_device;
                seed =
                    (static_cast<std::uint64_t>(random_device()) << 32) ^
                    static_cast<std::uint64_t>(random_device());
            }

            std::cerr << "random query seed: " << seed << "\n";
            std::mt19937_64 generator(seed);
            sources = generate_random_vertices(
                static_cast<std::size_t>(random_source_count),
                n,
                generator);
            targets = generate_random_vertices(
                static_cast<std::size_t>(random_target_count),
                n,
                generator);
        } else {
            if (seed_seen) {
                throw std::runtime_error(
                    "--seed is only valid with --random");
            }
            if (positional.size() != 2) {
                print_usage(argv[0]);
                return 1;
            }
            sources = read_vertex_id_file(
                positional[0],
                n,
                "source");
            targets = read_vertex_id_file(
                positional[1],
                n,
                "target");
        }

        const auto query_input_end = std::chrono::steady_clock::now();
        const double query_input_seconds =
            std::chrono::duration<double>(
                query_input_end - query_input_begin).count();

        std::cerr
            << std::fixed << std::setprecision(6)
            << "CH file read/parse time (excluded from benchmark): "
            << ch_read_seconds << " s\n"
            << "source/target input preparation (excluded from benchmark): "
            << query_input_seconds << " s\n";

        if (backend == "ch-parallel") {
            if (!parallel_ch_workspace) {
                throw std::runtime_error("Missing parallel CH persistent workspace");
            }
            std::cerr
                << "parallel CH persistent workspace init (excluded): "
                << parallel_ch_reusable_init_seconds << " s\n";
            run_many_to_many_parallel_ch(
                ch,
                *parallel_ch_workspace,
                sources,
                targets,
                quiet,
                requested_threads);
            return 0;
        }

#ifdef __CUDACC__
        if (backend == "cuda-g32-opt") {
            if (!g32_opt_host_workspace) {
                throw std::runtime_error("Missing optimized G32 persistent workspace");
            }
            run_many_to_many_cuda_g32_optimized(
                ch,
                *g32_opt_host_workspace,
                sources,
                targets,
                quiet,
                device_id,
                gpu_batch_sources,
                requested_threads,
                g32_opt_reusable_init_seconds);
            return 0;
        }
#endif

        const auto selection_start = std::chrono::steady_clock::now();
        RestrictedDownward restricted =
            build_restricted_downward(ch, targets);
        const auto selection_end = std::chrono::steady_clock::now();
        const double selection_seconds =
            std::chrono::duration<double>(
                selection_end - selection_start).count();

        const std::size_t downward_edge_count = ch.downward.num_edges();
        std::cerr
            << "CH: " << n << " vertices, "
            << ch.upward.num_edges() << " upward arcs, "
            << downward_edge_count << " downward arcs\n"
            << "RPHAST restricted graph: "
            << restricted.vertex.size() << " vertices, "
            << restricted.head.size() << " arcs\n";

        ch.downward = CSRGraph{};
        ch.reverse_downward = CSRGraph{};
        std::vector<std::uint32_t>().swap(ch.level);

        if (backend == "cpu" || backend == "cpu-rank") {
            run_many_to_many_cpu(
                ch,
                restricted,
                sources,
                targets,
                quiet,
                requested_threads,
                backend == "cpu"
                    ? CPUQueryMode::Dijkstra
                    : CPUQueryMode::RankSweep,
                selection_seconds);
        } else {
#ifdef __CUDACC__
            if (backend == "cuda-g32") {
                run_many_to_many_cuda_g32(
                    ch,
                    restricted,
                    sources,
                    targets,
                    quiet,
                    device_id,
                    gpu_batch_sources,
                    requested_threads,
                    selection_seconds);
            } else {
                run_many_to_many_cuda(
                    ch,
                    restricted,
                    sources,
                    targets,
                    quiet,
                    device_id,
                    gpu_batch_sources,
                    selection_seconds);
            }
#else
            throw std::runtime_error(
                "CUDA backend unavailable");
#endif
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 2;
    }
}

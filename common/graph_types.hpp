#pragma once

// Shared vertex/weight/graph types for the Hub Labeling and RPHAST query benchmarks.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using Vertex = std::uint32_t;
using StoredDistance = std::uint32_t;
using Weight = std::uint64_t;
using Offset = std::uint64_t;

static constexpr Weight INF = std::numeric_limits<Weight>::max() / 4;

static_assert(sizeof(void*) == 8, "64-bit build required");
static_assert(sizeof(Offset) >= 8, "64-bit graph offsets required");

struct Edge {
    Vertex to{};
    Weight weight{};
};

struct CSRGraph {
    std::vector<Offset> first;
    std::vector<Vertex> head;
    std::vector<Weight> weight;

    std::size_t num_vertices() const {
        return first.empty() ? 0 : first.size() - 1;
    }

    std::size_t num_edges() const {
        return head.size();
    }
};


class CSRBuilder {
public:
    CSRBuilder() = default;

    CSRBuilder(std::size_t n, std::size_t expected_edges)
        : first_(n + 1, 0), n_(n), expected_edges_(expected_edges) {
        head_.reserve(expected_edges);
        weight_.reserve(expected_edges);
    }

    void add_block(Vertex source, const std::vector<Edge>& edges) {
        if (source >= n_) {
            throw std::runtime_error("CSR source vertex is out of range");
        }
        if (source < next_source_) {
            throw std::runtime_error(
                "CSR blocks must be in increasing source order");
        }

        while (next_source_ < source) {
            first_[next_source_ + 1] = head_.size();
            ++next_source_;
        }

        for (const Edge& edge : edges) {
            if (edge.to >= n_) {
                throw std::runtime_error("CSR target vertex is out of range");
            }
            head_.push_back(edge.to);
            weight_.push_back(edge.weight);
        }

        first_[source + 1] = head_.size();
        next_source_ = source + 1;
    }

    CSRGraph finish() {
        while (next_source_ < n_) {
            first_[next_source_ + 1] = head_.size();
            ++next_source_;
        }
        if (head_.size() != expected_edges_) {
            throw std::runtime_error("CSR edge count does not match its header");
        }

        return CSRGraph{
            std::move(first_),
            std::move(head_),
            std::move(weight_)};
    }

private:
    std::vector<Offset> first_;
    std::vector<Vertex> head_;
    std::vector<Weight> weight_;
    std::size_t n_ = 0;
    std::size_t expected_edges_ = 0;
    Vertex next_source_ = 0;
};

inline CSRGraph reverse_graph(const CSRGraph& graph) {
    const std::size_t n = graph.num_vertices();
    std::vector<Offset> first(n + 1, 0);

    for (Vertex u = 0; u < n; ++u) {
        for (Offset i = graph.first[u]; i < graph.first[u + 1]; ++i) {
            ++first[static_cast<std::size_t>(graph.head[i]) + 1];
        }
    }
    for (std::size_t i = 1; i <= n; ++i) {
        first[i] += first[i - 1];
    }

    std::vector<Offset> cursor = first;
    std::vector<Vertex> head(graph.num_edges());
    std::vector<Weight> weight(graph.num_edges());

    for (Vertex u = 0; u < n; ++u) {
        for (Offset i = graph.first[u]; i < graph.first[u + 1]; ++i) {
            const Vertex v = graph.head[i];
            const Offset out = cursor[v]++;
            head[out] = u;
            weight[out] = graph.weight[i];
        }
    }

    return CSRGraph{
        std::move(first),
        std::move(head),
        std::move(weight)};
}

#pragma once

// Shared helpers for generating random benchmark queries

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "graph_types.hpp"

inline std::uint64_t mix_query_value(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

inline std::uint64_t distance_result_fingerprint(
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

    value ^= mix_query_value(distance_value);
    return mix_query_value(value);
}


inline std::vector<Vertex> generate_random_vertices(
    std::size_t count,
    std::size_t vertex_count,
    std::mt19937_64& generator) {

    if (count == 0) {
        throw std::runtime_error(
            "Random source/target counts must be positive");
    }

    if (count > vertex_count) {
        throw std::runtime_error(
            "Cannot choose more unique random vertices than the graph contains");
    }

    std::vector<Vertex> result;
    result.reserve(count);

    if (count > vertex_count / 4) {
        std::vector<Vertex> all(vertex_count);
        std::iota(all.begin(), all.end(), static_cast<Vertex>(0));
        std::shuffle(all.begin(), all.end(), generator);
        all.resize(count);
        return all;
    }

    std::unordered_set<Vertex> selected;

    if (count <= std::numeric_limits<std::size_t>::max() / 2) {
        selected.reserve(count * 2);
    }

    std::uniform_int_distribution<std::uint64_t> distribution(
        0,
        static_cast<std::uint64_t>(vertex_count - 1));

    while (result.size() < count) {
        const Vertex vertex = static_cast<Vertex>(distribution(generator));

        if (selected.insert(vertex).second) {
            result.push_back(vertex);
        }
    }

    return result;
}

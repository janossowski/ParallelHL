#pragma once

// Shared parsing/formatting helpers for the query benchmarks text I/O

#include <cstdint>
#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "graph_types.hpp"

inline bool next_data_line(std::istream& input, std::string& line) {
    while (std::getline(input, line)) {
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            continue;
        }
        if (line[first] == 'c') {
            continue;
        }
        return true;
    }
    return false;
}


inline std::vector<Vertex> read_vertex_id_file(
    const std::string& path,
    std::size_t vertex_count,
    const char* description) {

    std::ifstream input(path);

    if (!input) {
        throw std::runtime_error(
            std::string("Cannot open ") + description + " file: " + path);
    }

    std::vector<Vertex> vertices;
    std::string line;
    std::uint64_t line_number = 0;

    while (std::getline(input, line)) {
        ++line_number;
        const std::size_t first = line.find_first_not_of(" \t\r\n");

        if (first == std::string::npos ||
            line[first] == '#' ||
            line[first] == 'c') {
            continue;
        }

        std::istringstream stream(line);
        std::string token;

        while (stream >> token) {
            std::size_t parsed = 0;
            std::uint64_t id = 0;

            try {
                id = std::stoull(token, &parsed);
            } catch (const std::exception&) {
                throw std::runtime_error(
                    std::string("Invalid vertex ID in ") +
                    description +
                    " file at line " +
                    std::to_string(line_number) +
                    ": " + token);
            }

            if (parsed != token.size() || id == 0 || id > vertex_count) {
                throw std::runtime_error(
                    std::string("Vertex ID out of range in ") +
                    description +
                    " file at line " +
                    std::to_string(line_number) +
                    ": " + token);
            }

            vertices.push_back(static_cast<Vertex>(id - 1));
        }
    }

    if (!input.eof()) {
        throw std::runtime_error(
            std::string("Failed while reading ") +
            description +
            " file: " + path);
    }

    if (vertices.empty()) {
        throw std::runtime_error(
            std::string("No vertex IDs found in ") +
            description +
            " file: " + path);
    }

    return vertices;
}


inline std::string csv_escape(const std::string& value) {
    bool needs_quotes = false;
    for (char c : value) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            needs_quotes = true;
            break;
        }
    }
    if (!needs_quotes) {
        return value;
    }

    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char c : value) {
        if (c == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(c);
    }
    escaped.push_back('"');
    return escaped;
}

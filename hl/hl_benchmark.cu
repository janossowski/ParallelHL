#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
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

static_assert(sizeof(std::streamoff) >= 8, "64-bit file offsets required");

template <typename T>
static void write_items(
    std::ostream& output,
    const T* data,
    std::size_t count,
    const char* description) {

    if (count == 0) {
        return;
    }

    if (count >
        static_cast<std::size_t>(
            std::numeric_limits<std::streamsize>::max()) /
            sizeof(T)) {

        throw std::runtime_error(
            std::string("Binary write is too large for std::streamsize: ") +
            description);
    }

    const std::size_t byte_count =
        count * sizeof(T);

    errno = 0;

    output.write(
        reinterpret_cast<const char*>(data),
        static_cast<std::streamsize>(byte_count));

    if (!output) {
        std::ostringstream message;

        message
            << "Failed while writing "
            << description
            << "; requested "
            << byte_count
            << " bytes";

        if (errno != 0) {
            message
                << "; system error: "
                << std::strerror(errno)
                << " (errno " << errno << ")";
        }

        throw std::runtime_error(message.str());
    }
}


template <typename T>
static void read_items(
    std::istream& input,
    T* data,
    std::size_t count,
    const char* description) {

    if (count == 0) {
        return;
    }

    if (count >
        static_cast<std::size_t>(
            std::numeric_limits<std::streamsize>::max()) /
            sizeof(T)) {

        throw std::runtime_error(
            std::string("Binary read is too large for std::streamsize: ") +
            description);
    }

    const std::size_t byte_count = count * sizeof(T);

    input.read(
        reinterpret_cast<char*>(data),
        static_cast<std::streamsize>(byte_count));

    if (!input) {
        throw std::runtime_error(
            std::string("Failed while reading ") + description);
    }
}

static std::streamoff checked_stream_bytes(
    std::uint64_t count,
    std::uint64_t element_size,
    const char* description) {

    const auto maximum =
        static_cast<std::uint64_t>(
            std::numeric_limits<std::streamoff>::max());

    if (element_size != 0 && count > maximum / element_size) {
        throw std::runtime_error(
            std::string("File section is too large: ") + description);
    }

    return static_cast<std::streamoff>(count * element_size);
}

static void write_zeros(
    std::ostream& output,
    std::uint64_t byte_count) {

    constexpr std::size_t buffer_size = 1024 * 1024;

    std::vector<char> zeros(buffer_size, '\0');

    while (byte_count > 0) {
        const std::size_t chunk =
            static_cast<std::size_t>(
                std::min<std::uint64_t>(
                    byte_count,
                    static_cast<std::uint64_t>(zeros.size())));

        output.write(
            zeros.data(),
            static_cast<std::streamsize>(chunk));

        if (!output) {
            throw std::runtime_error(
                "Failed while reserving binary output space");
        }

        byte_count -= chunk;
    }
}

static void copy_file_to_stream(
    const std::filesystem::path& input_path,
    std::ostream& output) {

    std::ifstream input(input_path, std::ios::binary);

    if (!input) {
        throw std::runtime_error(
            "Cannot open temporary file: " + input_path.string());
    }

    constexpr std::size_t buffer_size = 4 * 1024 * 1024;

    std::vector<char> buffer(buffer_size);

    while (input) {
        input.read(
            buffer.data(),
            static_cast<std::streamsize>(buffer.size()));

        const std::streamsize count = input.gcount();

        if (count > 0) {
            output.write(buffer.data(), count);

            if (!output) {
                throw std::runtime_error(
                    "Failed while copying temporary distance data");
            }
        }
    }

    if (!input.eof()) {
        throw std::runtime_error(
            "Failed while reading temporary file: " +
            input_path.string());
    }
}

class TemporaryFileGuard {
public:
    explicit TemporaryFileGuard(std::filesystem::path path)
        : path_(std::move(path)) {}

    ~TemporaryFileGuard() {
        if (active_) {
            std::error_code error;
            std::filesystem::remove(path_, error);
        }
    }

    void release() {
        active_ = false;
    }

private:
    std::filesystem::path path_;
    bool active_ = true;
};


struct CHQueryGraph {
    std::vector<std::uint32_t> rank;
    std::vector<std::uint32_t> level;
    CSRGraph upward;
    CSRGraph reverse_downward;
};

static CHQueryGraph read_ch_text(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open CH file: " + path);

    std::string line;
    if (!next_data_line(in, line) || line != "CH_TEXT_V2") {
        throw std::runtime_error("Expected CH_TEXT_V2");
    }

    if (!next_data_line(in, line)) throw std::runtime_error("Missing CH header");
    std::istringstream hs(line);
    std::string p, ch, n_word, aug_word, up_word, down_word;
    std::uint64_t n64 = 0, augmented = 0, upward_edges = 0, downward_edges = 0;
    hs >> p >> ch >> n_word >> n64 >> aug_word >> augmented
       >> up_word >> upward_edges >> down_word >> downward_edges;
    if (!hs || p != "p" || ch != "ch" || n_word != "n" ||
        aug_word != "augmented_arcs" || up_word != "upward_edges" ||
        down_word != "downward_edges") {
        throw std::runtime_error("Malformed CH p-line");
    }
    if (n64 > std::numeric_limits<Vertex>::max()) {
        throw std::runtime_error("Too many vertices for 32-bit vertex IDs");
    }
    const std::size_t n = static_cast<std::size_t>(n64);

    std::vector<std::uint32_t> rank(n, std::numeric_limits<std::uint32_t>::max());
    std::vector<std::uint32_t> level(n, 0);
    bool got_upward = false;
    bool got_downward = false;
    CSRGraph upward;
    CSRGraph downward;

    while (next_data_line(in, line)) {
        std::istringstream ls(line);
        std::string token;
        ls >> token;

        if (token == "begin_vertices") {
            std::uint64_t count = 0;
            ls >> count;
            if (count != n64) throw std::runtime_error("Vertex count mismatch");

            std::uint64_t seen = 0;
            while (next_data_line(in, line)) {
                std::istringstream vs(line);
                std::string kind;
                vs >> kind;
                if (kind == "end_vertices") break;
                if (kind != "v") throw std::runtime_error("Expected v or end_vertices");

                std::uint64_t id = 0, r = 0, lev = 0;
                vs >> id >> r >> lev;
                if (!vs || id == 0 || id > n64 || r >= n64) {
                    throw std::runtime_error("Malformed vertex record");
                }
                rank[id - 1] = static_cast<std::uint32_t>(r);
                level[id - 1] = static_cast<std::uint32_t>(lev);
                ++seen;
            }
            if (seen != n64) throw std::runtime_error("Incomplete vertex section");
            continue;
        }

        if (token == "begin_augmented_arcs") {
            while (next_data_line(in, line)) {
                std::istringstream as(line);
                std::string t;
                as >> t;
                if (t == "end_augmented_arcs") break;
            }
            continue;
        }

        if (token == "begin_csr") {
            std::string direction, n_tag, edges_tag;
            std::uint64_t section_n = 0, section_edges = 0;
            ls >> direction >> n_tag >> section_n >> edges_tag >> section_edges;
            if (!ls || n_tag != "n" || edges_tag != "edges" || section_n != n64) {
                throw std::runtime_error("Malformed begin_csr line");
            }
            if (direction != "upward" && direction != "downward") {
                throw std::runtime_error("Unknown CSR direction");
            }

            CSRBuilder builder(n, static_cast<std::size_t>(section_edges));
            while (next_data_line(in, line)) {
                std::istringstream bs(line);
                std::string kind;
                bs >> kind;
                if (kind == "end_csr") {
                    std::string end_direction;
                    bs >> end_direction;
                    if (end_direction != direction) {
                        throw std::runtime_error("Mismatched end_csr direction");
                    }
                    break;
                }
                if (kind != "b") throw std::runtime_error("Expected b or end_csr");

                std::uint64_t source_id = 0, degree = 0;
                bs >> source_id >> degree;
                if (!bs || source_id == 0 || source_id > n64) {
                    throw std::runtime_error("Malformed b record");
                }

                std::vector<Edge> edges;
                edges.reserve(static_cast<std::size_t>(degree));
                for (std::uint64_t i = 0; i < degree; ++i) {
                    if (!next_data_line(in, line)) {
                        throw std::runtime_error("Unexpected EOF inside CSR block");
                    }
                    std::istringstream es(line);
                    std::string e_kind;
                    std::uint64_t target_id = 0, w = 0;
                    es >> e_kind >> target_id >> w;
                    if (!es || e_kind != "e" || target_id == 0 || target_id > n64) {
                        throw std::runtime_error("Malformed e record");
                    }
                    edges.push_back(Edge{static_cast<Vertex>(target_id - 1), w});
                }
                builder.add_block(static_cast<Vertex>(source_id - 1), edges);
            }

            CSRGraph graph = builder.finish();
            if (direction == "upward") {
                if (section_edges != upward_edges) {
                    throw std::runtime_error("Upward edge count disagrees with p-line");
                }
                upward = std::move(graph);
                got_upward = true;
            } else {
                if (section_edges != downward_edges) {
                    throw std::runtime_error("Downward edge count disagrees with p-line");
                }
                downward = std::move(graph);
                got_downward = true;
            }
            continue;
        }

        if (token == "end_ch") break;
    }

    if (!got_upward || !got_downward) {
        throw std::runtime_error("CH file must contain upward and downward CSR sections");
    }
    for (std::size_t v = 0; v < n; ++v) {
        if (rank[v] == std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("Missing rank for a vertex");
        }
    }

    for (Vertex u = 0; u < n; ++u) {
        for (Offset i = upward.first[u]; i < upward.first[u + 1]; ++i) {
            if (!(rank[u] < rank[upward.head[i]])) {
                throw std::runtime_error("Invalid upward edge rank orientation");
            }
        }
        for (Offset i = downward.first[u]; i < downward.first[u + 1]; ++i) {
            if (!(rank[u] > rank[downward.head[i]])) {
                throw std::runtime_error("Invalid downward edge rank orientation");
            }
        }
    }

    return CHQueryGraph{
        std::move(rank),
        std::move(level),
        std::move(upward),
        reverse_graph(downward)
    };
}

struct LabelEntry {
    Vertex hub{};
    Weight distance{};
};

enum class PruningMode {
    None,
    OneHop,
    TwoHop
};

static const char* pruning_mode_name(PruningMode mode) {
    switch (mode) {
        case PruningMode::None:
            return "none";

        case PruningMode::OneHop:
            return "1-hop";

        case PruningMode::TwoHop:
            return "2-hop";
    }

    return "unknown";
}

static unsigned pruning_hops(PruningMode mode) {
    switch (mode) {
        case PruningMode::None:
            return 0;

        case PruningMode::OneHop:
            return 1;

        case PruningMode::TwoHop:
            return 2;
    }

    return 0;
}


struct SearchStats {
    std::uint64_t scanned = 0;

    std::uint64_t stalled = 0;
    std::uint64_t stalled_depth1 = 0;
    std::uint64_t stalled_depth2 = 0;
};

struct StoreBuildStats {
    std::uint64_t entries = 0;
    SearchStats search;
    Weight maximum_distance = 0;
};

class DijkstraWorkspace {
public:
    explicit DijkstraWorkspace(std::size_t n)
        : dist_(n, INF) {}

    std::vector<LabelEntry> run_with_pruning(
        const CSRGraph& search_graph,
        const CSRGraph& stall_graph,
        Vertex source,
        unsigned max_stall_hops,
        SearchStats* stats = nullptr) {

        if (search_graph.num_vertices() != dist_.size() ||
            stall_graph.num_vertices() != dist_.size()) {

            throw std::runtime_error(
                "Dijkstra graph size mismatch");
        }

        if (max_stall_hops > 2) {
            throw std::runtime_error(
                "This implementation supports at most 2-hop pruning");
        }

        for (Vertex v : touched_) {
            dist_[v] = INF;
        }

        touched_.clear();

        using QueueEntry =
            std::pair<Weight, Vertex>;

        std::priority_queue<
            QueueEntry,
            std::vector<QueueEntry>,
            std::greater<QueueEntry>
        > queue;

        dist_[source] = 0;
        touched_.push_back(source);
        queue.emplace(0, source);

        std::vector<LabelEntry> result;

        while (!queue.empty()) {
            const auto [du, u] = queue.top();
            queue.pop();


            if (du != dist_[u]) {
                continue;
            }

            unsigned stall_depth = 0;

            if (max_stall_hops != 0) {
                stall_depth =
                    find_stall_depth(
                        stall_graph,
                        u,
                        du,
                        max_stall_hops);
            }

            if (stall_depth != 0) {
                if (stats != nullptr) {
                    ++stats->stalled;

                    if (stall_depth == 1) {
                        ++stats->stalled_depth1;
                    } else if (stall_depth == 2) {
                        ++stats->stalled_depth2;
                    }
                }

                continue;
            }

            if (stats != nullptr) {
                ++stats->scanned;
            }

            result.push_back(
                LabelEntry{u, du});


            for (Offset i = search_graph.first[u];
                 i < search_graph.first[u + 1];
                 ++i) {

                const Vertex v =
                    search_graph.head[i];

                const Weight edge_weight =
                    search_graph.weight[i];

                if (edge_weight >= INF) {
                    continue;
                }

                if (du > INF - edge_weight) {
                    continue;
                }

                const Weight candidate =
                    du + edge_weight;

                if (candidate < dist_[v]) {
                    if (dist_[v] == INF) {
                        touched_.push_back(v);
                    }

                    dist_[v] = candidate;
                    queue.emplace(candidate, v);
                }
            }
        }

        std::sort(
            result.begin(),
            result.end(),
            [](const LabelEntry& a,
               const LabelEntry& b) {

                return a.hub < b.hub;
            });

        return result;
    }

private:
    static bool witness_improves(
        Weight witness_distance,
        Weight path_distance,
        Weight current_distance) {

        if (witness_distance == INF) {
            return false;
        }

        if (path_distance >= INF) {
            return false;
        }

        if (witness_distance > INF - path_distance) {
            return false;
        }

        return
            witness_distance + path_distance <
            current_distance;
    }

    unsigned find_stall_depth(
        const CSRGraph& stall_graph,
        Vertex w,
        Weight dw,
        unsigned max_hops) const {

        for (Offset first_edge = stall_graph.first[w];
             first_edge < stall_graph.first[w + 1];
             ++first_edge) {

            const Vertex x =
                stall_graph.head[first_edge];

            const Weight wx =
                stall_graph.weight[first_edge];

            if (witness_improves(
                    dist_[x],
                    wx,
                    dw)) {

                return 1;
            }
        }

        if (max_hops < 2) {
            return 0;
        }

        for (Offset first_edge = stall_graph.first[w];
             first_edge < stall_graph.first[w + 1];
             ++first_edge) {

            const Vertex x =
                stall_graph.head[first_edge];

            const Weight wx =
                stall_graph.weight[first_edge];

            if (wx >= INF) {
                continue;
            }

            for (Offset second_edge = stall_graph.first[x];
                 second_edge < stall_graph.first[x + 1];
                 ++second_edge) {

                const Vertex witness =
                    stall_graph.head[second_edge];

                const Weight xu =
                    stall_graph.weight[second_edge];

                if (xu >= INF || wx > INF - xu) {
                    continue;
                }

                const Weight path_distance = wx + xu;

                if (witness_improves(
                        dist_[witness],
                        path_distance,
                        dw)) {

                    return 2;
                }
            }
        }

        return 0;
    }

    std::vector<Weight> dist_;
    std::vector<Vertex> touched_;
};

static StoreBuildStats build_and_write_store(
    std::ofstream& output,
    const CSRGraph& search_graph,
    const CSRGraph& stall_graph,
    DijkstraWorkspace& workspace,
    const std::filesystem::path& distance_temp_path,
    const char* direction_name,
    PruningMode pruning_mode) {

    const std::size_t n = search_graph.num_vertices();

    if (stall_graph.num_vertices() != n) {
        throw std::runtime_error(
            "Search and stall graph sizes differ");
    }

    std::vector<Offset> offsets(n + 1, 0);

    const std::streampos store_begin = output.tellp();

    if (store_begin == std::streampos(-1)) {
        throw std::runtime_error(
            "Cannot determine output position");
    }

    const std::uint64_t placeholder_entries = 0;

    write_items(
        output,
        &placeholder_entries,
        1,
        "placeholder entry count");

    write_zeros(
        output,
        static_cast<std::uint64_t>(n + 1) *
            static_cast<std::uint64_t>(sizeof(Offset)));

    TemporaryFileGuard distance_guard(distance_temp_path);

    std::ofstream distance_output(
        distance_temp_path,
        std::ios::binary | std::ios::trunc);

    if (!distance_output) {
        throw std::runtime_error(
            "Cannot create temporary distance file: " +
            distance_temp_path.string());
    }

    std::vector<Vertex> hub_buffer;
    std::vector<StoredDistance> distance_buffer;

    StoreBuildStats stats;

    for (std::size_t source_index = 0;
         source_index < n;
         ++source_index) {

        const Vertex source =
            static_cast<Vertex>(source_index);

        std::vector<LabelEntry> label =
            workspace.run_with_pruning(
                search_graph,
                stall_graph,
                source,
                pruning_hops(pruning_mode),
                &stats.search);

        hub_buffer.resize(label.size());
        distance_buffer.resize(label.size());

        for (std::size_t i = 0; i < label.size(); ++i) {
            hub_buffer[i] = label[i].hub;

            const Weight distance = label[i].distance;

            stats.maximum_distance =
                std::max(stats.maximum_distance, distance);
                
            if (distance >
                static_cast<Weight>(
                    std::numeric_limits<StoredDistance>::max())) {

                throw std::runtime_error(
                    "Label distance does not fit in 32 bits: source " +
                    std::to_string(static_cast<std::uint64_t>(source) + 1) +
                    ", hub " +
                    std::to_string(
                        static_cast<std::uint64_t>(label[i].hub) + 1) +
                    ", distance " +
                    std::to_string(distance));
            }

            distance_buffer[i] =
                static_cast<StoredDistance>(distance);
        }

        write_items(
            output,
            hub_buffer.data(),
            hub_buffer.size(),
            "hub IDs to the partial label file");

        write_items(
            distance_output,
            distance_buffer.data(),
            distance_buffer.size(),
            "distances to the temporary distance file");

        stats.entries +=
            static_cast<std::uint64_t>(label.size());

        offsets[source_index + 1] =
            static_cast<Offset>(stats.entries);

        if ((source_index + 1) % 100000 == 0 || source_index + 1 == n) {
            const std::uint64_t processed =
                static_cast<std::uint64_t>(
                    source_index + 1);

            const std::uint64_t scanned =
                stats.search.scanned;

            const std::uint64_t stalled =
                stats.search.stalled;

            const std::uint64_t considered =
                scanned + stalled;

            const double average_size =
                processed == 0
                    ? 0.0
                    : static_cast<double>(stats.entries) /
                        static_cast<double>(processed);

            const double direct_stall_rate =
                considered == 0
                    ? 0.0
                    : 100.0 *
                        static_cast<double>(stalled) /
                        static_cast<double>(considered);

            if (scanned != stats.entries) {
                throw std::runtime_error(
                    "Internal statistics error: scanned != entries");
            }

            std::cerr
                << direction_name
                << " labels: "
                << processed << "/" << n
                << ", pruning: "
                << pruning_mode_name(pruning_mode)
                << ", entries: "
                << stats.entries
                << ", average label: "
                << std::fixed
                << std::setprecision(2)
                << average_size
                << ", scanned: "
                << scanned;

            if (pruning_mode != PruningMode::None) {
                std::cerr
                    << ", stalled: "
                    << stalled
                    << ", direct stall rate: "
                    << direct_stall_rate
                    << "%"
                    << ", 1-hop stalls: "
                    << stats.search.stalled_depth1;

                if (pruning_mode == PruningMode::TwoHop) {
                    std::cerr
                        << ", additional 2-hop stalls: "
                        << stats.search.stalled_depth2;
                }
            }

            std::cerr
                << ", maximum label distance: "
                << stats.maximum_distance
                << "\n";
        }
    }

    const std::uint64_t expected_distance_bytes =
        stats.entries *
        static_cast<std::uint64_t>(sizeof(StoredDistance));

    std::cerr
        << direction_name
        << ": flushing temporary distance file; expected size: "
        << expected_distance_bytes
        << " bytes\n";

    distance_output.flush();

    if (distance_output.fail()) {
        const std::filesystem::path directory =
            distance_temp_path.parent_path().empty()
                ? std::filesystem::current_path()
                : distance_temp_path.parent_path();

        std::error_code space_error;

        const std::filesystem::space_info space =
            std::filesystem::space(
                directory,
                space_error);

        std::ostringstream message;

        message
            << "Failed while flushing temporary distance file '"
            << distance_temp_path.string()
            << "'; expected size: "
            << expected_distance_bytes
            << " bytes";

        if (!space_error) {
            message
                << "; filesystem available space: "
                << space.available
                << " bytes";
        } else {
            message
                << "; could not determine available space: "
                << space_error.message();
        }

        throw std::runtime_error(message.str());
    }

    std::cerr
        << direction_name
        << ": closing temporary distance file\n";

    distance_output.close();

    if (distance_output.fail()) {
        throw std::runtime_error(
            "Failed while closing temporary distance file '" +
            distance_temp_path.string() + "'");
    }

    std::error_code file_size_error;

    const std::uintmax_t actual_distance_bytes =
        std::filesystem::file_size(
            distance_temp_path,
            file_size_error);

    if (file_size_error) {
        throw std::runtime_error(
            "Cannot determine temporary distance file size: " +
            file_size_error.message());
    }

    if (actual_distance_bytes != expected_distance_bytes) {
        throw std::runtime_error(
            "Temporary distance file has unexpected size: expected " +
            std::to_string(expected_distance_bytes) +
            " bytes, got " +
            std::to_string(actual_distance_bytes) +
            " bytes");
    }

    std::cerr
        << direction_name
        << ": appending "
        << actual_distance_bytes
        << " distance bytes\n";

    copy_file_to_stream(
        distance_temp_path,
        output);

    if (!output) {
        throw std::runtime_error(
            "Failed while appending temporary distance data");
    }

    const std::streampos store_end = output.tellp();

    if (store_end == std::streampos(-1)) {
        throw std::runtime_error(
            "Cannot determine final store position");
    }

    std::cerr
        << direction_name
        << ": backpatching offsets\n";

    output.seekp(store_begin);

    if (!output) {
        throw std::runtime_error(
            "Failed to seek while backpatching store");
    }

    write_items(
        output,
        &stats.entries,
        1,
        "final entry count");

    write_items(
        output,
        offsets.data(),
        offsets.size(),
        "label offsets");

    output.seekp(store_end);

    if (!output) {
        throw std::runtime_error(
            "Failed to restore output position after backpatching");
    }

    std::error_code remove_error;

    const bool removed =
        std::filesystem::remove(
            distance_temp_path,
            remove_error);

    if (remove_error) {
        throw std::runtime_error(
            "Failed to remove temporary distance file '" +
            distance_temp_path.string() +
            "': " +
            remove_error.message());
    }

    if (!removed) {
        throw std::runtime_error(
            "Temporary distance file disappeared before cleanup: " +
            distance_temp_path.string());
    }

    distance_guard.release();

    return stats;
}

static void build_labels_streaming(
    const CHQueryGraph& ch,
    const std::string& output_path_string,
    PruningMode pruning_mode) {

    const std::size_t n = ch.upward.num_vertices();

    if (ch.reverse_downward.num_vertices() != n) {
        throw std::runtime_error(
            "Forward and reverse CH graph sizes differ");
    }
    std::cerr
        << "HL pruning mode: "
        << pruning_mode_name(pruning_mode)
        << "\n";

    const std::filesystem::path output_path(output_path_string);
    const std::filesystem::path partial_path =
        output_path.string() + ".partial";

    const std::filesystem::path forward_distance_temp =
        output_path.string() + ".forward.dist.tmp";

    const std::filesystem::path reverse_distance_temp =
        output_path.string() + ".reverse.dist.tmp";

    TemporaryFileGuard partial_guard(partial_path);

    std::ofstream output(
        partial_path,
        std::ios::binary | std::ios::trunc);

    if (!output) {
        throw std::runtime_error(
            "Cannot create label file: " + partial_path.string());
    }

    const char magic[8] = {
        'H', 'L', 'B', 'A', 'S', 'E', '2', '\0'
    };

    write_items(
        output,
        magic,
        sizeof(magic),
        "HL file magic");

    const std::uint64_t n64 =
        static_cast<std::uint64_t>(n);

    write_items(
        output,
        &n64,
        1,
        "HL vertex count");

    DijkstraWorkspace workspace(n);

    const StoreBuildStats forward_stats =
        build_and_write_store(
            output,
            ch.upward,
            ch.reverse_downward,
            workspace,
            forward_distance_temp,
            "forward",
            pruning_mode);

    const StoreBuildStats reverse_stats =
        build_and_write_store(
            output,
            ch.reverse_downward,
            ch.upward,
            workspace,
            reverse_distance_temp,
            "reverse",
            pruning_mode);

    output.close();

    if (!output) {
        throw std::runtime_error(
            "Failed while closing label output file");
    }

    std::error_code filesystem_error;

    std::filesystem::remove(output_path, filesystem_error);
    filesystem_error.clear();

    std::filesystem::rename(
        partial_path,
        output_path,
        filesystem_error);

    if (filesystem_error) {
        throw std::runtime_error(
            "Cannot rename completed label file: " +
            filesystem_error.message());
    }

    partial_guard.release();

    const double average_forward =
        n == 0
            ? 0.0
            : static_cast<double>(forward_stats.entries) /
                  static_cast<double>(n);

    const double average_reverse =
        n == 0
            ? 0.0
            : static_cast<double>(reverse_stats.entries) /
                  static_cast<double>(n);

    const double average_all =
        n == 0
            ? 0.0
            : (static_cast<double>(forward_stats.entries) +
               static_cast<double>(reverse_stats.entries)) /
                  (2.0 * static_cast<double>(n));

    std::cerr
        << "\nHL build statistics\n"
        << "pruning: "
        << pruning_mode_name(pruning_mode)
        << "\n"
        << "forward entries: "
        << forward_stats.entries
        << "\n"
        << "reverse entries: "
        << reverse_stats.entries
        << "\n"
        << "forward scanned: "
        << forward_stats.search.scanned
        << "\n"
        << "reverse scanned: "
        << reverse_stats.search.scanned
        << "\n";

    if (pruning_mode != PruningMode::None) {
        std::cerr
            << "forward total stalls: "
            << forward_stats.search.stalled
            << "\n"
            << "reverse total stalls: "
            << reverse_stats.search.stalled
            << "\n"
            << "forward 1-hop stalls: "
            << forward_stats.search.stalled_depth1
            << "\n"
            << "reverse 1-hop stalls: "
            << reverse_stats.search.stalled_depth1
            << "\n";

        if (pruning_mode == PruningMode::TwoHop) {
            std::cerr
                << "forward additional 2-hop stalls: "
                << forward_stats.search.stalled_depth2
                << "\n"
                << "reverse additional 2-hop stalls: "
                << reverse_stats.search.stalled_depth2
                << "\n";
        }
    }

    std::cerr
        << std::fixed
        << std::setprecision(2)
        << "average forward label size: "
        << average_forward
        << "\n"
        << "average reverse label size: "
        << average_reverse
        << "\n"
        << "average label size: "
        << average_all
        << "\n";

}

struct QueryResult {
    Weight distance = INF;
    Vertex hub = std::numeric_limits<Vertex>::max();
};

struct DiskLabelStore {
    std::uint64_t entries = 0;
    std::vector<Offset> offset;
    std::streamoff hub_begin = 0;
    std::streamoff distance_begin = 0;
};

struct DiskHubLabels {
    std::ifstream input;
    std::size_t num_vertices = 0;
    DiskLabelStore forward;
    DiskLabelStore reverse;
};

struct LoadedLabel {
    std::vector<Vertex> hub;
    std::vector<StoredDistance> distance;
};

struct ResidentLabelStore {
    std::vector<Offset> offset;
    std::vector<Vertex> hub;
    std::vector<StoredDistance> distance;
};

struct ResidentHubLabels {
    std::size_t num_vertices = 0;
    ResidentLabelStore forward;
    ResidentLabelStore reverse;
};

template <typename T>
static void read_large_array_at(
    std::ifstream& input,
    std::streamoff begin,
    std::uint64_t count64,
    std::vector<T>& output,
    const char* description) {

    if (count64 >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(
            std::string(description) + " does not fit in size_t");
    }

    output.resize(static_cast<std::size_t>(count64));

    input.clear();
    input.seekg(begin, std::ios::beg);
    if (!input) {
        throw std::runtime_error(
            std::string("Failed to seek to resident ") + description);
    }

    constexpr std::size_t chunk_bytes = 64ULL * 1024ULL * 1024ULL;
    constexpr std::size_t chunk_elements =
        chunk_bytes / sizeof(T) == 0 ? 1 : chunk_bytes / sizeof(T);

    std::size_t position = 0;
    while (position < output.size()) {
        const std::size_t count =
            std::min(chunk_elements, output.size() - position);
        read_items(
            input,
            output.data() + position,
            count,
            description);
        position += count;
    }
}

static ResidentHubLabels load_labels_resident(
    DiskHubLabels& disk) {

    ResidentHubLabels resident;
    resident.num_vertices = disk.num_vertices;

    resident.forward.offset = std::move(disk.forward.offset);
    resident.reverse.offset = std::move(disk.reverse.offset);

    read_large_array_at(
        disk.input,
        disk.forward.hub_begin,
        disk.forward.entries,
        resident.forward.hub,
        "forward resident hubs");
    read_large_array_at(
        disk.input,
        disk.forward.distance_begin,
        disk.forward.entries,
        resident.forward.distance,
        "forward resident distances");
    read_large_array_at(
        disk.input,
        disk.reverse.hub_begin,
        disk.reverse.entries,
        resident.reverse.hub,
        "reverse resident hubs");
    read_large_array_at(
        disk.input,
        disk.reverse.distance_begin,
        disk.reverse.entries,
        resident.reverse.distance,
        "reverse resident distances");

    return resident;
}

static std::uint64_t estimate_resident_label_bytes(
    const DiskHubLabels& disk) {

    auto checked_add = [](std::uint64_t a, std::uint64_t b) {
        if (a > std::numeric_limits<std::uint64_t>::max() - b) {
            throw std::runtime_error(
                "Resident HL byte estimate overflows uint64_t");
        }
        return a + b;
    };

    const std::uint64_t offset_bytes =
        static_cast<std::uint64_t>(disk.num_vertices + 1) *
        sizeof(Offset);

    const std::uint64_t forward_entry_bytes =
        disk.forward.entries *
        (sizeof(Vertex) + sizeof(StoredDistance));
    const std::uint64_t reverse_entry_bytes =
        disk.reverse.entries *
        (sizeof(Vertex) + sizeof(StoredDistance));

    std::uint64_t total = 0;
    total = checked_add(total, offset_bytes);
    total = checked_add(total, offset_bytes);
    total = checked_add(total, forward_entry_bytes);
    total = checked_add(total, reverse_entry_bytes);
    return total;
}

static DiskLabelStore read_disk_store_index(
    std::ifstream& input,
    std::size_t vertex_count,
    const char* direction_name) {

    DiskLabelStore store;

    read_items(
        input,
        &store.entries,
        1,
        direction_name);

    store.offset.resize(vertex_count + 1);

    read_items(
        input,
        store.offset.data(),
        store.offset.size(),
        direction_name);

    if (store.offset.empty() ||
        store.offset.front() != 0 ||
        store.offset.back() != store.entries) {

        throw std::runtime_error(
            std::string("Malformed ") +
            direction_name +
            " label offsets");
    }

    for (std::size_t i = 1; i < store.offset.size(); ++i) {
        if (store.offset[i] < store.offset[i - 1]) {
            throw std::runtime_error(
                std::string("Non-monotonic ") +
                direction_name +
                " label offsets");
        }
    }

    const std::streampos hub_position = input.tellg();

    if (hub_position == std::streampos(-1)) {
        throw std::runtime_error(
            std::string("Cannot locate ") +
            direction_name +
            " hub array");
    }

    store.hub_begin = static_cast<std::streamoff>(hub_position);

    input.seekg(
        checked_stream_bytes(
            store.entries,
            sizeof(Vertex),
            "hub array"),
        std::ios::cur);

    if (!input) {
        throw std::runtime_error(
            std::string("Failed to skip ") +
            direction_name +
            " hub array");
    }

    const std::streampos distance_position = input.tellg();

    if (distance_position == std::streampos(-1)) {
        throw std::runtime_error(
            std::string("Cannot locate ") +
            direction_name +
            " distance array");
    }

    store.distance_begin =
        static_cast<std::streamoff>(distance_position);

    input.seekg(
        checked_stream_bytes(
            store.entries,
            sizeof(StoredDistance),
            "distance array"),
        std::ios::cur);

    if (!input) {
        throw std::runtime_error(
            std::string("Failed to skip ") +
            direction_name +
            " distance array");
    }

    return store;
}

static DiskHubLabels open_labels_for_query(
    const std::string& path) {

    DiskHubLabels labels;

    labels.input.open(path, std::ios::binary);

    if (!labels.input) {
        throw std::runtime_error(
            "Cannot open label file: " + path);
    }

    char magic[8]{};

    read_items(
        labels.input,
        magic,
        sizeof(magic),
        "label file magic");

    const char expected[8] = {
        'H', 'L', 'B', 'A', 'S', 'E', '2', '\0'
    };

    if (std::memcmp(
            magic,
            expected,
            sizeof(expected)) != 0) {

        throw std::runtime_error(
            "Not an HLBASE2 label file. Rebuild labels with this version "
            "if the file was produced by an older or mislabeled writer.");
    }

    std::uint64_t vertex_count64 = 0;

    read_items(
        labels.input,
        &vertex_count64,
        1,
        "HL vertex count");

    if (vertex_count64 >
        static_cast<std::uint64_t>(
            std::numeric_limits<Vertex>::max())) {

        throw std::runtime_error(
            "Vertex count does not fit in Vertex");
    }

    if (vertex_count64 >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {

        throw std::runtime_error(
            "Vertex count does not fit in size_t");
    }

    labels.num_vertices =
        static_cast<std::size_t>(vertex_count64);

    labels.forward =
        read_disk_store_index(
            labels.input,
            labels.num_vertices,
            "forward store");

    labels.reverse =
        read_disk_store_index(
            labels.input,
            labels.num_vertices,
            "reverse store");

    return labels;
}

static LoadedLabel read_label_from_disk(
    std::ifstream& input,
    const DiskLabelStore& store,
    Vertex vertex,
    std::size_t vertex_count,
    const char* description) {

    if (static_cast<std::size_t>(vertex) >= vertex_count) {
        throw std::runtime_error(
            std::string("Invalid vertex for ") + description);
    }

    const Offset begin = store.offset[vertex];
    const Offset end = store.offset[
        static_cast<std::size_t>(vertex) + 1];

    if (begin > end || end > store.entries) {
        throw std::runtime_error(
            std::string("Malformed offsets for ") + description);
    }

    const std::uint64_t count64 =
        static_cast<std::uint64_t>(end - begin);

    if (count64 >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {

        throw std::runtime_error(
            std::string("Label is too large for memory: ") +
            description);
    }

    const std::size_t count =
        static_cast<std::size_t>(count64);

    LoadedLabel label;
    label.hub.resize(count);
    label.distance.resize(count);

    const std::streamoff hub_delta =
        checked_stream_bytes(
            static_cast<std::uint64_t>(begin),
            sizeof(Vertex),
            "hub label position");

    if (store.hub_begin >
        std::numeric_limits<std::streamoff>::max() - hub_delta) {

        throw std::runtime_error(
            "Hub label file position overflows streamoff");
    }

    input.clear();
    input.seekg(store.hub_begin + hub_delta, std::ios::beg);

    if (!input) {
        throw std::runtime_error(
            std::string("Failed to seek to ") +
            description +
            " hubs");
    }

    read_items(
        input,
        label.hub.data(),
        label.hub.size(),
        description);

    const std::streamoff distance_delta =
        checked_stream_bytes(
            static_cast<std::uint64_t>(begin),
            sizeof(StoredDistance),
            "distance label position");

    if (store.distance_begin >
        std::numeric_limits<std::streamoff>::max() - distance_delta) {

        throw std::runtime_error(
            "Distance label file position overflows streamoff");
    }

    input.clear();
    input.seekg(
        store.distance_begin + distance_delta,
        std::ios::beg);

    if (!input) {
        throw std::runtime_error(
            std::string("Failed to seek to ") +
            description +
            " distances");
    }

    read_items(
        input,
        label.distance.data(),
        label.distance.size(),
        description);

    return label;
}

static QueryResult intersect_labels(
    const LoadedLabel& forward,
    const LoadedLabel& reverse) {

    if (forward.hub.size() != forward.distance.size() ||
        reverse.hub.size() != reverse.distance.size()) {

        throw std::runtime_error(
            "Loaded label hub/distance arrays have different sizes");
    }

    QueryResult best;

    std::size_t i = 0;
    std::size_t j = 0;

    while (i < forward.hub.size() &&
           j < reverse.hub.size()) {

        const Vertex a = forward.hub[i];
        const Vertex b = reverse.hub[j];

        if (a < b) {
            ++i;
            continue;
        }

        if (b < a) {
            ++j;
            continue;
        }

        const Weight left =
            static_cast<Weight>(forward.distance[i]);

        const Weight right =
            static_cast<Weight>(reverse.distance[j]);

        if (left <= INF - right) {
            const Weight candidate = left + right;

            if (candidate < best.distance) {
                best.distance = candidate;
                best.hub = a;
            }
        }

        ++i;
        ++j;
    }

    return best;
}

static std::uint64_t label_entry_count(
    const DiskLabelStore& store,
    Vertex vertex) {

    return static_cast<std::uint64_t>(
        store.offset[static_cast<std::size_t>(vertex) + 1] -
        store.offset[vertex]);
}

static std::uint64_t estimated_loaded_label_bytes(
    const DiskLabelStore& store,
    Vertex vertex) {

    const std::uint64_t entries =
        label_entry_count(store, vertex);

    constexpr std::uint64_t per_entry =
        sizeof(Vertex) + sizeof(StoredDistance);

    const std::uint64_t maximum =
        std::numeric_limits<std::uint64_t>::max();

    if (entries > maximum / per_entry) {
        return maximum;
    }

    const std::uint64_t data_bytes = entries * per_entry;
    constexpr std::uint64_t overhead = 64;

    if (data_bytes > maximum - overhead) {
        return maximum;
    }

    return data_bytes + overhead;
}

static std::uint64_t estimate_selected_label_bytes(
    const DiskLabelStore& store,
    const std::vector<Vertex>& vertices) {

    std::uint64_t total = 0;

    for (Vertex vertex : vertices) {
        const std::uint64_t bytes =
            estimated_loaded_label_bytes(store, vertex);

        if (total >
            std::numeric_limits<std::uint64_t>::max() - bytes) {

            return std::numeric_limits<std::uint64_t>::max();
        }

        total += bytes;
    }

    return total;
}

static std::uint64_t estimate_selected_label_bytes(
    const ResidentLabelStore& store,
    const std::vector<Vertex>& vertices) {

    std::uint64_t total = 0;
    constexpr std::uint64_t per_entry =
        sizeof(Vertex) + sizeof(StoredDistance);
    constexpr std::uint64_t overhead = 64;

    for (Vertex vertex : vertices) {
        const std::uint64_t entries =
            static_cast<std::uint64_t>(
                store.offset[static_cast<std::size_t>(vertex) + 1] -
                store.offset[vertex]);

        if (entries >
            (std::numeric_limits<std::uint64_t>::max() - overhead) /
                per_entry) {
            return std::numeric_limits<std::uint64_t>::max();
        }

        const std::uint64_t bytes =
            entries * per_entry + overhead;

        if (total >
            std::numeric_limits<std::uint64_t>::max() - bytes) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        total += bytes;
    }

    return total;
}

static std::uint64_t query_result_fingerprint(
    Vertex source,
    Vertex target,
    const QueryResult& result) {

    const std::uint64_t distance_value =
        result.distance == INF
            ? 0xffffffffffffffffULL
            : static_cast<std::uint64_t>(result.distance);

    const std::uint64_t hub_value =
        result.hub == std::numeric_limits<Vertex>::max()
            ? 0xffffffffULL
            : static_cast<std::uint64_t>(result.hub);

    std::uint64_t value =
        (static_cast<std::uint64_t>(source) << 32) ^
        static_cast<std::uint64_t>(target);

    value ^= mix_query_value(distance_value);
    value ^= mix_query_value(hub_value + 0x632be59bd9b4e019ULL);

    return mix_query_value(value);
}

static void print_query_result(
    Vertex source,
    Vertex target,
    const QueryResult& result) {

    std::cout
        << (static_cast<std::uint64_t>(source) + 1)
        << ' '
        << (static_cast<std::uint64_t>(target) + 1)
        << ' ';

    if (result.distance == INF) {
        std::cout << "unreachable 0\n";
    } else {
        std::cout
            << result.distance
            << ' '
            << (static_cast<std::uint64_t>(result.hub) + 1)
            << '\n';
    }
}

static void run_many_to_many_queries_cpu(
    DiskHubLabels& labels,
    const std::vector<Vertex>& sources,
    const std::vector<Vertex>& targets,
    std::uint64_t cache_limit_bytes,
    bool quiet,
    int requested_threads) {

    if (cache_limit_bytes == 0) {
        throw std::runtime_error(
            "Query cache size must be greater than zero");
    }

    if (sources.empty() || targets.empty()) {
        throw std::runtime_error(
            "Source and target sets must both be non-empty");
    }

    if (requested_threads <= 0) {
        throw std::runtime_error(
            "Thread count must be positive");
    }

#ifndef _OPENMP
    if (requested_threads != 1) {
        throw std::runtime_error(
            "This executable was built without OpenMP support. "
            "Recompile with -fopenmp (GCC/Clang) or /openmp (MSVC), "
            "or use --threads 1.");
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

    const std::uint64_t source_bytes =
        estimate_selected_label_bytes(labels.forward, sources);

    const std::uint64_t target_bytes =
        estimate_selected_label_bytes(labels.reverse, targets);

    if (source_bytes == std::numeric_limits<std::uint64_t>::max() ||
        target_bytes == std::numeric_limits<std::uint64_t>::max() ||
        source_bytes >
            std::numeric_limits<std::uint64_t>::max() - target_bytes) {

        throw std::runtime_error(
            "Estimated memory required by selected labels overflows uint64_t");
    }

    const std::uint64_t total_label_bytes =
        source_bytes + target_bytes;

    std::cerr
        << "many-to-many query: "
        << sources.size()
        << " sources x "
        << targets.size()
        << " targets = "
        << pair_count
        << " pairs\n"
        << "estimated preloaded label memory: "
        << std::fixed
        << std::setprecision(2)
        << (total_label_bytes / (1024.0 * 1024.0))
        << " MiB, limit: "
        << (cache_limit_bytes / (1024.0 * 1024.0))
        << " MiB\n";

    if (total_label_bytes > cache_limit_bytes) {
        std::ostringstream message;
        message
            << "All required labels need approximately "
            << std::fixed
            << std::setprecision(2)
            << (total_label_bytes / (1024.0 * 1024.0))
            << " MiB, exceeding --cache-mb "
            << (cache_limit_bytes / (1024ULL * 1024ULL))
            << ". Increase --cache-mb so every required label can be "
               "preloaded before timing.";
        throw std::runtime_error(message.str());
    }

    /*
     * Load every forward and reverse label needed by the benchmark before
     * starting the timer.  The timed region performs no .hlbin I/O.
     */
    const auto preload_start =
        std::chrono::steady_clock::now();

    std::vector<LoadedLabel> source_labels;
    source_labels.reserve(sources.size());

    for (Vertex source : sources) {
        source_labels.push_back(
            read_label_from_disk(
                labels.input,
                labels.forward,
                source,
                labels.num_vertices,
                "forward label"));
    }

    std::vector<LoadedLabel> target_labels;
    target_labels.reserve(targets.size());

    for (Vertex target : targets) {
        target_labels.push_back(
            read_label_from_disk(
                labels.input,
                labels.reverse,
                target,
                labels.num_vertices,
                "reverse label"));
    }

    const auto preload_end =
        std::chrono::steady_clock::now();

    const double preload_seconds =
        std::chrono::duration<double>(
            preload_end - preload_start).count();

    std::cerr
        << "labels preloaded; .hlbin I/O time (not benchmarked): "
        << std::fixed
        << std::setprecision(6)
        << preload_seconds
        << " s\n";

    std::vector<QueryResult> stored_results;

    if (!quiet) {
        if (pair_count >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error(
                "Too many query results to store for non-quiet output; use --quiet");
        }

        const std::size_t result_count =
            static_cast<std::size_t>(pair_count);

        if (result_count > stored_results.max_size()) {
            throw std::runtime_error(
                "Too many query results to store; use --quiet");
        }

        stored_results.resize(result_count);
    }


    // Initialize the OpenMP runtime before the benchmark timer

    int actual_threads = 1;

#ifdef _OPENMP
    omp_set_dynamic(0);

#pragma omp parallel num_threads(requested_threads)
    {
#pragma omp single
        {
            actual_threads = omp_get_num_threads();
        }
    }
#else
    actual_threads = 1;
#endif

    std::cerr
        << "query workers: "
        << actual_threads
#ifdef _OPENMP
        << " (OpenMP)\n";
#else
        << " (serial; OpenMP not enabled)\n";
#endif

    std::vector<std::uint64_t> worker_checksums(
        static_cast<std::size_t>(actual_threads), 0);
    std::vector<std::uint64_t> worker_unreachable(
        static_cast<std::size_t>(actual_threads), 0);

#ifdef _OPENMP
    const std::uint64_t target_count =
        static_cast<std::uint64_t>(targets.size());
#endif

    const auto query_start =
        std::chrono::steady_clock::now();

#ifdef _OPENMP
#pragma omp parallel num_threads(requested_threads)
    {
        const int thread_id = omp_get_thread_num();
        const int thread_count = omp_get_num_threads();

        const std::uint64_t thread_id64 =
            static_cast<std::uint64_t>(thread_id);
        const std::uint64_t thread_count64 =
            static_cast<std::uint64_t>(thread_count);

        const std::uint64_t base =
            pair_count / thread_count64;
        const std::uint64_t remainder =
            pair_count % thread_count64;

        const std::uint64_t begin =
            thread_id64 * base +
            std::min(thread_id64, remainder);

        const std::uint64_t local_count =
            base + (thread_id64 < remainder ? 1ULL : 0ULL);

        const std::uint64_t end =
            begin + local_count;

        std::uint64_t local_checksum = 0;
        std::uint64_t local_unreachable = 0;

        if (begin < end) {
            std::uint64_t source_index =
                begin / target_count;
            std::uint64_t target_index =
                begin % target_count;

            for (std::uint64_t pair_index = begin;
                 pair_index < end;
                 ++pair_index) {

                const LoadedLabel& forward =
                    source_labels[static_cast<std::size_t>(source_index)];

                const QueryResult result =
                    intersect_labels(
                        forward,
                        target_labels[
                            static_cast<std::size_t>(target_index)]);

                local_checksum ^=
                    query_result_fingerprint(
                        sources[static_cast<std::size_t>(source_index)],
                        targets[static_cast<std::size_t>(target_index)],
                        result);

                if (result.distance == INF) {
                    ++local_unreachable;
                }

                if (!quiet) {
                    stored_results[
                        static_cast<std::size_t>(pair_index)] = result;
                }

                ++target_index;
                if (target_index == target_count) {
                    target_index = 0;
                    ++source_index;
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
        std::uint64_t pair_index = 0;

        for (std::size_t source_index = 0;
             source_index < sources.size();
             ++source_index) {

            const LoadedLabel& forward =
                source_labels[source_index];

            for (std::size_t target_index = 0;
                 target_index < targets.size();
                 ++target_index) {

                const QueryResult result =
                    intersect_labels(
                        forward,
                        target_labels[target_index]);

                local_checksum ^=
                    query_result_fingerprint(
                        sources[source_index],
                        targets[target_index],
                        result);

                if (result.distance == INF) {
                    ++local_unreachable;
                }

                if (!quiet) {
                    stored_results[
                        static_cast<std::size_t>(pair_index)] = result;
                }

                ++pair_index;
            }
        }

        worker_checksums[0] = local_checksum;
        worker_unreachable[0] = local_unreachable;
    }
#endif

    const auto query_end =
        std::chrono::steady_clock::now();

    std::uint64_t checksum = 0;
    std::uint64_t unreachable_count = 0;

    for (std::size_t i = 0; i < worker_checksums.size(); ++i) {
        checksum ^= worker_checksums[i];
        unreachable_count += worker_unreachable[i];
    }

    const double query_seconds =
        std::chrono::duration<double>(
            query_end - query_start).count();

    const double total_nanoseconds =
        std::chrono::duration<double, std::nano>(
            query_end - query_start).count();

    const double nanoseconds_per_pair =
        total_nanoseconds / static_cast<double>(pair_count);

    const double pairs_per_second =
        query_seconds > 0.0
            ? static_cast<double>(pair_count) / query_seconds
            : 0.0;

    std::cerr
        << "timed parallel query computation complete\n"
        << "query compute time: "
        << std::fixed
        << std::setprecision(6)
        << query_seconds
        << " s\n"
        << "average wall time: "
        << std::setprecision(2)
        << nanoseconds_per_pair
        << " ns/pair\n"
        << "throughput: "
        << std::setprecision(2)
        << pairs_per_second
        << " pairs/s\n"
        << "query pairs completed: "
        << pair_count
        << ", unreachable: "
        << unreachable_count
        << ", checksum: "
        << checksum
        << "\n";

    if (!quiet) {
        std::size_t index = 0;
        for (Vertex source : sources) {
            for (Vertex target : targets) {
                print_query_result(
                    source,
                    target,
                    stored_results[index]);
                ++index;
            }
        }
    }
}


#ifdef __CUDACC__


// CUDA query backend.

struct FlatLabelSet {
    std::vector<Offset> offset;
    std::vector<Vertex> hub;
    std::vector<StoredDistance> distance;
};

struct DeviceQueryResult {
    Weight distance;
    Vertex hub;
};


static FlatLabelSet preload_flat_label_set(
    std::ifstream& input,
    const DiskLabelStore& store,
    const std::vector<Vertex>& vertices,
    std::size_t vertex_count,
    const char* description) {

    std::uint64_t total_entries64 = 0;

    for (Vertex vertex : vertices) {
        if (static_cast<std::size_t>(vertex) >= vertex_count) {
            throw std::runtime_error(
                std::string("Invalid vertex while preloading ") + description);
        }

        const Offset begin = store.offset[vertex];
        const Offset end = store.offset[static_cast<std::size_t>(vertex) + 1];

        if (begin > end || end > store.entries) {
            throw std::runtime_error(
                std::string("Malformed offsets while preloading ") + description);
        }

        const std::uint64_t count =
            static_cast<std::uint64_t>(end - begin);

        if (total_entries64 >
            std::numeric_limits<std::uint64_t>::max() - count) {
            throw std::runtime_error(
                std::string("Selected ") + description +
                " entry count overflows uint64_t");
        }

        total_entries64 += count;
    }

    if (total_entries64 >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(
            std::string("Selected ") + description +
            " entries do not fit in size_t");
    }

    FlatLabelSet flat;
    flat.offset.reserve(vertices.size() + 1);
    flat.offset.push_back(0);

    const std::size_t total_entries =
        static_cast<std::size_t>(total_entries64);

    flat.hub.reserve(total_entries);
    flat.distance.reserve(total_entries);

    for (Vertex vertex : vertices) {
        LoadedLabel label =
            read_label_from_disk(
                input,
                store,
                vertex,
                vertex_count,
                description);

        if (flat.hub.size() >
            std::numeric_limits<Offset>::max() - label.hub.size()) {
            throw std::runtime_error(
                std::string("Flattened ") + description +
                " offsets overflow Offset");
        }

        flat.hub.insert(
            flat.hub.end(),
            label.hub.begin(),
            label.hub.end());

        flat.distance.insert(
            flat.distance.end(),
            label.distance.begin(),
            label.distance.end());

        flat.offset.push_back(
            static_cast<Offset>(flat.hub.size()));
    }

    return flat;
}

static FlatLabelSet preload_flat_label_set_resident(
    const ResidentLabelStore& store,
    const std::vector<Vertex>& vertices,
    std::size_t vertex_count,
    const char* description) {

    std::uint64_t total_entries64 = 0;

    for (Vertex vertex : vertices) {
        if (static_cast<std::size_t>(vertex) >= vertex_count) {
            throw std::runtime_error(
                std::string("Invalid vertex while staging resident ") +
                description);
        }

        const Offset begin = store.offset[vertex];
        const Offset end = store.offset[static_cast<std::size_t>(vertex) + 1];

        if (begin > end ||
            end > static_cast<Offset>(store.hub.size())) {
            throw std::runtime_error(
                std::string("Malformed resident offsets while staging ") +
                description);
        }

        total_entries64 += static_cast<std::uint64_t>(end - begin);
    }

    if (total_entries64 >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(
            std::string("Selected resident ") + description +
            " entries do not fit in size_t");
    }

    FlatLabelSet flat;
    flat.offset.reserve(vertices.size() + 1);
    flat.offset.push_back(0);
    flat.hub.reserve(static_cast<std::size_t>(total_entries64));
    flat.distance.reserve(static_cast<std::size_t>(total_entries64));

    for (Vertex vertex : vertices) {
        const Offset begin = store.offset[vertex];
        const Offset end = store.offset[static_cast<std::size_t>(vertex) + 1];

        const auto hub_begin = store.hub.begin() +
            static_cast<std::ptrdiff_t>(begin);
        const auto hub_end = store.hub.begin() +
            static_cast<std::ptrdiff_t>(end);
        const auto distance_begin = store.distance.begin() +
            static_cast<std::ptrdiff_t>(begin);
        const auto distance_end = store.distance.begin() +
            static_cast<std::ptrdiff_t>(end);

        flat.hub.insert(flat.hub.end(), hub_begin, hub_end);
        flat.distance.insert(
            flat.distance.end(),
            distance_begin,
            distance_end);
        flat.offset.push_back(static_cast<Offset>(flat.hub.size()));
    }

    return flat;
}


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

    CudaBuffer(CudaBuffer&& other) noexcept
        : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                cudaFree(data_);
            }
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void allocate(std::size_t count) {
        if (data_ != nullptr) {
            throw std::runtime_error("CUDA buffer allocated twice");
        }

        count_ = count;
        if (count_ == 0) {
            return;
        }

        if (count_ >
            std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::runtime_error("CUDA allocation byte count overflows size_t");
        }

        cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&data_),
                count_ * sizeof(T)),
            "cudaMalloc");
    }

    void upload(const std::vector<T>& host) {
        if (host.size() != count_) {
            throw std::runtime_error("CUDA upload size mismatch");
        }
        if (count_ == 0) {
            return;
        }

        cuda_check(
            cudaMemcpy(
                data_,
                host.data(),
                count_ * sizeof(T),
                cudaMemcpyHostToDevice),
            "cudaMemcpy host-to-device");
    }

    void download(std::vector<T>& host) const {
        host.resize(count_);
        if (count_ == 0) {
            return;
        }

        cuda_check(
            cudaMemcpy(
                host.data(),
                data_,
                count_ * sizeof(T),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy device-to-host");
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t size() const { return count_; }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

__device__ __forceinline__ unsigned long long cuda_mix_query_value(
    unsigned long long value) {

    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__device__ __forceinline__ unsigned long long cuda_distance_fingerprint(
    Vertex source,
    Vertex target,
    Weight distance) {

    const unsigned long long distance_value =
        distance == INF
            ? 0xffffffffffffffffULL
            : static_cast<unsigned long long>(distance);

    unsigned long long value =
        (static_cast<unsigned long long>(source) << 32) ^
        static_cast<unsigned long long>(target);

    value ^= cuda_mix_query_value(distance_value);
    return cuda_mix_query_value(value);
}

__device__ __forceinline__ unsigned long long cuda_query_fingerprint(
    Vertex source,
    Vertex target,
    Weight distance,
    Vertex hub) {

    const unsigned long long distance_value =
        distance == INF
            ? 0xffffffffffffffffULL
            : static_cast<unsigned long long>(distance);

    const unsigned long long hub_value =
        hub == 0xffffffffU
            ? 0xffffffffULL
            : static_cast<unsigned long long>(hub);

    unsigned long long value =
        (static_cast<unsigned long long>(source) << 32) ^
        static_cast<unsigned long long>(target);

    value ^= cuda_mix_query_value(distance_value);
    value ^= cuda_mix_query_value(
        hub_value + 0x632be59bd9b4e019ULL);

    return cuda_mix_query_value(value);
}

__device__ __forceinline__ DeviceQueryResult cuda_intersect_labels(
    const Offset* forward_offset,
    const Vertex* forward_hub,
    const StoredDistance* forward_distance,
    std::uint64_t source_index,
    const Offset* reverse_offset,
    const Vertex* reverse_hub,
    const StoredDistance* reverse_distance,
    std::uint64_t target_index) {

    Offset i = forward_offset[source_index];
    const Offset i_end = forward_offset[source_index + 1];

    Offset j = reverse_offset[target_index];
    const Offset j_end = reverse_offset[target_index + 1];

    Weight best_distance = INF;
    Vertex best_hub = 0xffffffffU;

    while (i < i_end && j < j_end) {
        const Vertex a = forward_hub[i];
        const Vertex b = reverse_hub[j];

        if (a < b) {
            ++i;
            continue;
        }

        if (b < a) {
            ++j;
            continue;
        }

        const Weight left =
            static_cast<Weight>(forward_distance[i]);
        const Weight right =
            static_cast<Weight>(reverse_distance[j]);

        if (left <= INF - right) {
            const Weight candidate = left + right;

            if (candidate < best_distance) {
                best_distance = candidate;
                best_hub = a;
            }
        }

        ++i;
        ++j;
    }

    DeviceQueryResult result;
    result.distance = best_distance;
    result.hub = best_hub;
    return result;
}

/*
 * One CUDA thread owns one s-t pair.  Pairs are flattened in source-major
 * order, matching the OpenMP implementation.  This lets threads of a given warp
 * to often share the same source label when the target set is at least a warp
 * wide, which gives the source side cache reuse.
 */
__global__ void hub_label_query_kernel(
    const Offset* source_offset,
    const Vertex* source_hub,
    const StoredDistance* source_distance,
    const Offset* target_offset,
    const Vertex* target_hub,
    const StoredDistance* target_distance,
    const Vertex* source_ids,
    const Vertex* target_ids,
    std::uint64_t source_count,
    std::uint64_t target_count,
    std::uint64_t pair_count,
    DeviceQueryResult* results,
    unsigned long long* block_checksums,
    unsigned long long* block_unreachable) {

    extern __shared__ unsigned long long shared[];
    unsigned long long* shared_checksum = shared;
    unsigned long long* shared_unreachable =
        shared + blockDim.x;

    const std::uint64_t pair_index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    unsigned long long local_checksum = 0;
    unsigned long long local_unreachable = 0;

    if (pair_index < pair_count) {
        const std::uint64_t source_index =
            pair_index / target_count;
        const std::uint64_t target_index =
            pair_index - source_index * target_count;

        const DeviceQueryResult result =
            cuda_intersect_labels(
                source_offset,
                source_hub,
                source_distance,
                source_index,
                target_offset,
                target_hub,
                target_distance,
                target_index);

        if (results != nullptr) {
            results[pair_index] = result;
        }

        local_checksum =
            results == nullptr
                ? cuda_distance_fingerprint(
                    source_ids[source_index],
                    target_ids[target_index],
                    result.distance)
                : cuda_query_fingerprint(
                    source_ids[source_index],
                    target_ids[target_index],
                    result.distance,
                    result.hub);

        local_unreachable =
            result.distance == INF ? 1ULL : 0ULL;
    }

    shared_checksum[threadIdx.x] = local_checksum;
    shared_unreachable[threadIdx.x] = local_unreachable;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2;
         stride != 0;
         stride >>= 1) {

        if (threadIdx.x < stride) {
            shared_checksum[threadIdx.x] ^=
                shared_checksum[threadIdx.x + stride];

            shared_unreachable[threadIdx.x] +=
                shared_unreachable[threadIdx.x + stride];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_checksums[blockIdx.x] = shared_checksum[0];
        block_unreachable[blockIdx.x] = shared_unreachable[0];
    }
}


/*
 * Hub-centric CUDA query backends.
 *
 * hub-source:
 *   one CUDA thread owns one actual (hub, source-label occurrence) and loops
 *   over all selected target-label occurrences containing that hub.
 *
 * hub-triple:
 *   one CUDA block owns one actual (hub, source-label occurrence).  Threads in
 *   the block cooperatively traverse that hub's contiguous target bucket, so
 *   each active thread performs a (hub, source, target) relaxation.
 */
struct TargetHubOccurrence {
    std::uint32_t target_index;
    StoredDistance distance;
};

struct HubSourceWork {
    Offset target_begin;
    std::uint32_t target_count;
    std::uint32_t source_index;
    StoredDistance source_distance;
    Vertex hub;
};

struct HubBucketWorkspace {
    std::vector<std::uint32_t> stamp;
    std::vector<std::uint32_t> bucket_id;
    std::uint32_t epoch = 0;

    explicit HubBucketWorkspace(std::size_t vertex_count)
        : stamp(vertex_count, 0),
          bucket_id(vertex_count, 0) {}

    std::uint32_t begin_query() {
        ++epoch;
        if (epoch == 0) {
            std::fill(stamp.begin(), stamp.end(), 0);
            epoch = 1;
        }
        return epoch;
    }
};

struct HubInvertedWork {
    std::vector<TargetHubOccurrence> target_occurrences;
    std::vector<HubSourceWork> source_work;
    std::uint64_t common_hubs = 0;
    std::uint64_t candidate_relaxations = 0;
    std::uint64_t maximum_target_fanout = 0;
    Weight maximum_candidate_distance = 0;
    std::uint64_t target_bucket_count = 0;
    std::uint64_t peak_query_host_bytes = 0;
};

static std::uint64_t checked_host_capacity_bytes(
    std::size_t capacity,
    std::size_t element_size,
    const char* description) {

    if (element_size != 0 &&
        capacity > std::numeric_limits<std::uint64_t>::max() / element_size) {
        throw std::runtime_error(
            std::string("Host byte count overflow for ") + description);
    }

    return static_cast<std::uint64_t>(capacity) * element_size;
}

template <class T>
static void add_host_vector_capacity_bytes(
    std::uint64_t& total,
    const std::vector<T>& values,
    const char* description) {

    const std::uint64_t bytes =
        checked_host_capacity_bytes(values.capacity(), sizeof(T), description);
    if (total > std::numeric_limits<std::uint64_t>::max() - bytes) {
        throw std::runtime_error(
            std::string("Host byte total overflow for ") + description);
    }
    total += bytes;
}

// Query-time no-sort bucket construction.

static HubInvertedWork build_hub_bucket_work_no_sort(
    const FlatLabelSet& source_labels,
    const FlatLabelSet& target_labels,
    HubBucketWorkspace& workspace) {

    if (source_labels.offset.empty() || target_labels.offset.empty()) {
        throw std::runtime_error(
            "Missing flat-label offsets while building hub buckets");
    }

    if (source_labels.hub.size() != source_labels.distance.size() ||
        target_labels.hub.size() != target_labels.distance.size()) {
        throw std::runtime_error(
            "Hub/distance size mismatch while building hub buckets");
    }

    const std::size_t source_count = source_labels.offset.size() - 1;
    const std::size_t target_count = target_labels.offset.size() - 1;

    if (source_count > std::numeric_limits<std::uint32_t>::max() ||
        target_count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "Hub-centric CUDA query counts must fit in uint32_t");
    }

    const std::uint32_t epoch = workspace.begin_query();

    std::vector<std::uint32_t> bucket_counts;
    std::vector<StoredDistance> bucket_max_target_distance;

    const std::size_t bucket_reserve =
        std::min(target_labels.hub.size(), workspace.stamp.size());
    bucket_counts.reserve(bucket_reserve);
    bucket_max_target_distance.reserve(bucket_reserve);

    for (std::size_t i = 0; i < target_labels.hub.size(); ++i) {
        const Vertex hub = target_labels.hub[i];

        if (static_cast<std::size_t>(hub) >= workspace.stamp.size()) {
            throw std::runtime_error(
                "Hub ID is out of range while building target buckets");
        }

        std::uint32_t bucket = 0;

        if (workspace.stamp[hub] != epoch) {
            if (bucket_counts.size() >=
                static_cast<std::size_t>(
                    std::numeric_limits<std::uint32_t>::max())) {
                throw std::runtime_error(
                    "Too many target buckets for uint32_t bucket IDs");
            }

            workspace.stamp[hub] = epoch;
            bucket = static_cast<std::uint32_t>(bucket_counts.size());
            workspace.bucket_id[hub] = bucket;
            bucket_counts.push_back(0);
            bucket_max_target_distance.push_back(0);
        } else {
            bucket = workspace.bucket_id[hub];
        }

        if (bucket_counts[bucket] ==
            std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error(
                "A target hub bucket exceeds uint32_t capacity");
        }

        ++bucket_counts[bucket];
        bucket_max_target_distance[bucket] =
            std::max(
                bucket_max_target_distance[bucket],
                target_labels.distance[i]);
    }

    std::vector<Offset> bucket_begin(bucket_counts.size() + 1, 0);

    for (std::size_t b = 0; b < bucket_counts.size(); ++b) {
        if (bucket_begin[b] >
            std::numeric_limits<Offset>::max() - bucket_counts[b]) {
            throw std::runtime_error(
                "Hub target bucket offsets overflow Offset");
        }
        bucket_begin[b + 1] = bucket_begin[b] + bucket_counts[b];
    }

    if (bucket_begin.back() !=
        static_cast<Offset>(target_labels.hub.size())) {
        throw std::runtime_error(
            "Internal error while counting target bucket entries");
    }

    HubInvertedWork work;
    work.target_bucket_count = bucket_counts.size();
    work.target_occurrences.resize(target_labels.hub.size());

    std::vector<Offset> cursor(
        bucket_begin.begin(),
        bucket_begin.end() - 1);

    for (std::size_t target_index = 0;
         target_index < target_count;
         ++target_index) {

        const Offset begin = target_labels.offset[target_index];
        const Offset end = target_labels.offset[target_index + 1];

        if (begin > end ||
            end > static_cast<Offset>(target_labels.hub.size())) {
            throw std::runtime_error(
                "Malformed target flat-label offsets while building buckets");
        }

        for (Offset i = begin; i < end; ++i) {
            const std::size_t position = static_cast<std::size_t>(i);
            const Vertex hub = target_labels.hub[position];
            const std::uint32_t bucket = workspace.bucket_id[hub];
            const Offset out = cursor[bucket]++;

            work.target_occurrences[static_cast<std::size_t>(out)] =
                TargetHubOccurrence{
                    static_cast<std::uint32_t>(target_index),
                    target_labels.distance[position]};
        }
    }

    std::vector<std::uint8_t> bucket_used(bucket_counts.size(), 0);
    work.source_work.reserve(source_labels.hub.size());

    for (std::size_t source_index = 0;
         source_index < source_count;
         ++source_index) {

        const Offset begin = source_labels.offset[source_index];
        const Offset end = source_labels.offset[source_index + 1];

        if (begin > end ||
            end > static_cast<Offset>(source_labels.hub.size())) {
            throw std::runtime_error(
                "Malformed source flat-label offsets while building work");
        }

        for (Offset i = begin; i < end; ++i) {
            const std::size_t position = static_cast<std::size_t>(i);
            const Vertex hub = source_labels.hub[position];

            if (static_cast<std::size_t>(hub) >= workspace.stamp.size() ||
                workspace.stamp[hub] != epoch) {
                continue;
            }

            const std::uint32_t bucket = workspace.bucket_id[hub];
            const std::uint32_t fanout = bucket_counts[bucket];

            if (fanout == 0) {
                continue;
            }

            if (!bucket_used[bucket]) {
                bucket_used[bucket] = 1;
                ++work.common_hubs;
                work.maximum_target_fanout =
                    std::max(
                        work.maximum_target_fanout,
                        static_cast<std::uint64_t>(fanout));
            }

            if (work.candidate_relaxations >
                std::numeric_limits<std::uint64_t>::max() - fanout) {
                throw std::runtime_error(
                    "Hub-centric candidate count overflows uint64_t");
            }
            work.candidate_relaxations += fanout;

            const Weight candidate_upper_bound =
                static_cast<Weight>(source_labels.distance[position]) +
                static_cast<Weight>(bucket_max_target_distance[bucket]);

            work.maximum_candidate_distance =
                std::max(
                    work.maximum_candidate_distance,
                    candidate_upper_bound);

            work.source_work.push_back(
                HubSourceWork{
                    bucket_begin[bucket],
                    fanout,
                    static_cast<std::uint32_t>(source_index),
                    source_labels.distance[position],
                    hub});
        }
    }

    add_host_vector_capacity_bytes(
        work.peak_query_host_bytes, bucket_counts, "bucket counts");
    add_host_vector_capacity_bytes(
        work.peak_query_host_bytes, bucket_max_target_distance,
        "bucket maximum target distances");
    add_host_vector_capacity_bytes(
        work.peak_query_host_bytes, bucket_begin, "bucket begin offsets");
    add_host_vector_capacity_bytes(
        work.peak_query_host_bytes, cursor, "bucket fill cursors");
    add_host_vector_capacity_bytes(
        work.peak_query_host_bytes, bucket_used, "bucket-used flags");
    add_host_vector_capacity_bytes(
        work.peak_query_host_bytes, work.target_occurrences,
        "target occurrences");
    add_host_vector_capacity_bytes(
        work.peak_query_host_bytes, work.source_work, "hub-source work");

    return work;
}

static HubInvertedWork build_hub_bucket_work_no_sort_resident(
    const ResidentLabelStore& forward,
    const std::vector<Vertex>& sources,
    const ResidentLabelStore& reverse,
    const std::vector<Vertex>& targets,
    HubBucketWorkspace& workspace) {

    if (forward.hub.size() != forward.distance.size() ||
        reverse.hub.size() != reverse.distance.size()) {
        throw std::runtime_error(
            "Resident hub/distance arrays have different sizes");
    }

    if (sources.size() > std::numeric_limits<std::uint32_t>::max() ||
        targets.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "Hub-centric CUDA query counts must fit in uint32_t");
    }

    std::uint64_t target_entries64 = 0;
    std::uint64_t source_entries64 = 0;

    for (Vertex target : targets) {
        const Offset begin = reverse.offset[target];
        const Offset end = reverse.offset[static_cast<std::size_t>(target) + 1];
        if (begin > end || end > reverse.hub.size()) {
            throw std::runtime_error("Malformed resident reverse offsets");
        }
        target_entries64 += static_cast<std::uint64_t>(end - begin);
    }

    for (Vertex source : sources) {
        const Offset begin = forward.offset[source];
        const Offset end = forward.offset[static_cast<std::size_t>(source) + 1];
        if (begin > end || end > forward.hub.size()) {
            throw std::runtime_error("Malformed resident forward offsets");
        }
        source_entries64 += static_cast<std::uint64_t>(end - begin);
    }

    if (target_entries64 >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max()) ||
        source_entries64 >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(
            "Selected resident label entries do not fit in size_t");
    }

    const std::uint32_t epoch = workspace.begin_query();

    std::vector<std::uint32_t> bucket_counts;
    std::vector<StoredDistance> bucket_max_target_distance;

    const std::size_t bucket_reserve =
        std::min(
            static_cast<std::size_t>(target_entries64),
            workspace.stamp.size());
    bucket_counts.reserve(bucket_reserve);
    bucket_max_target_distance.reserve(bucket_reserve);

    for (Vertex target : targets) {
        const Offset begin = reverse.offset[target];
        const Offset end = reverse.offset[static_cast<std::size_t>(target) + 1];

        for (Offset i = begin; i < end; ++i) {
            const std::size_t position = static_cast<std::size_t>(i);
            const Vertex hub = reverse.hub[position];

            if (static_cast<std::size_t>(hub) >= workspace.stamp.size()) {
                throw std::runtime_error(
                    "Resident reverse hub ID is out of range");
            }

            std::uint32_t bucket = 0;
            if (workspace.stamp[hub] != epoch) {
                if (bucket_counts.size() >=
                    static_cast<std::size_t>(
                        std::numeric_limits<std::uint32_t>::max())) {
                    throw std::runtime_error(
                        "Too many target buckets for uint32_t IDs");
                }

                workspace.stamp[hub] = epoch;
                bucket = static_cast<std::uint32_t>(bucket_counts.size());
                workspace.bucket_id[hub] = bucket;
                bucket_counts.push_back(0);
                bucket_max_target_distance.push_back(0);
            } else {
                bucket = workspace.bucket_id[hub];
            }

            if (bucket_counts[bucket] ==
                std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    "A target hub bucket exceeds uint32_t capacity");
            }

            ++bucket_counts[bucket];
            bucket_max_target_distance[bucket] =
                std::max(
                    bucket_max_target_distance[bucket],
                    reverse.distance[position]);
        }
    }

    std::vector<Offset> bucket_begin(bucket_counts.size() + 1, 0);
    for (std::size_t b = 0; b < bucket_counts.size(); ++b) {
        if (bucket_begin[b] >
            std::numeric_limits<Offset>::max() - bucket_counts[b]) {
            throw std::runtime_error(
                "Hub target bucket offsets overflow Offset");
        }
        bucket_begin[b + 1] = bucket_begin[b] + bucket_counts[b];
    }

    if (bucket_begin.back() != target_entries64) {
        throw std::runtime_error(
            "Internal error while counting resident target buckets");
    }

    HubInvertedWork work;
    work.target_bucket_count = bucket_counts.size();
    work.target_occurrences.resize(
        static_cast<std::size_t>(target_entries64));

    std::vector<Offset> cursor(
        bucket_begin.begin(),
        bucket_begin.end() - 1);

    for (std::size_t target_index = 0;
         target_index < targets.size();
         ++target_index) {

        const Vertex target_vertex = targets[target_index];
        const Offset begin = reverse.offset[target_vertex];
        const Offset end =
            reverse.offset[static_cast<std::size_t>(target_vertex) + 1];

        for (Offset i = begin; i < end; ++i) {
            const std::size_t position = static_cast<std::size_t>(i);
            const Vertex hub = reverse.hub[position];
            const std::uint32_t bucket = workspace.bucket_id[hub];
            const Offset out = cursor[bucket]++;

            work.target_occurrences[static_cast<std::size_t>(out)] =
                TargetHubOccurrence{
                    static_cast<std::uint32_t>(target_index),
                    reverse.distance[position]};
        }
    }

    std::vector<std::uint8_t> bucket_used(bucket_counts.size(), 0);
    work.source_work.reserve(
        static_cast<std::size_t>(source_entries64));

    for (std::size_t source_index = 0;
         source_index < sources.size();
         ++source_index) {

        const Vertex source_vertex = sources[source_index];
        const Offset begin = forward.offset[source_vertex];
        const Offset end =
            forward.offset[static_cast<std::size_t>(source_vertex) + 1];

        for (Offset i = begin; i < end; ++i) {
            const std::size_t position = static_cast<std::size_t>(i);
            const Vertex hub = forward.hub[position];

            if (static_cast<std::size_t>(hub) >= workspace.stamp.size() ||
                workspace.stamp[hub] != epoch) {
                continue;
            }

            const std::uint32_t bucket = workspace.bucket_id[hub];
            const std::uint32_t fanout = bucket_counts[bucket];

            if (!bucket_used[bucket]) {
                bucket_used[bucket] = 1;
                ++work.common_hubs;
                work.maximum_target_fanout =
                    std::max(
                        work.maximum_target_fanout,
                        static_cast<std::uint64_t>(fanout));
            }

            if (work.candidate_relaxations >
                std::numeric_limits<std::uint64_t>::max() - fanout) {
                throw std::runtime_error(
                    "Hub-centric candidate count overflows uint64_t");
            }
            work.candidate_relaxations += fanout;

            const Weight candidate_upper_bound =
                static_cast<Weight>(forward.distance[position]) +
                static_cast<Weight>(bucket_max_target_distance[bucket]);
            work.maximum_candidate_distance =
                std::max(
                    work.maximum_candidate_distance,
                    candidate_upper_bound);

            work.source_work.push_back(
                HubSourceWork{
                    bucket_begin[bucket],
                    fanout,
                    static_cast<std::uint32_t>(source_index),
                    forward.distance[position],
                    hub});
        }
    }

    add_host_vector_capacity_bytes(
        work.peak_query_host_bytes, bucket_counts, "resident bucket counts");
    add_host_vector_capacity_bytes(
        work.peak_query_host_bytes, bucket_max_target_distance,
        "resident bucket maximum target distances");
    add_host_vector_capacity_bytes(
        work.peak_query_host_bytes, bucket_begin,
        "resident bucket begin offsets");
    add_host_vector_capacity_bytes(
        work.peak_query_host_bytes, cursor, "resident bucket fill cursors");
    add_host_vector_capacity_bytes(
        work.peak_query_host_bytes, bucket_used, "resident bucket-used flags");
    add_host_vector_capacity_bytes(
        work.peak_query_host_bytes, work.target_occurrences,
        "resident target occurrences");
    add_host_vector_capacity_bytes(
        work.peak_query_host_bytes, work.source_work,
        "resident hub-source work");

    return work;
}

__device__ __forceinline__ unsigned long long cuda_pack_hub_result(
    StoredDistance distance,
    Vertex hub) {

    return
        (static_cast<unsigned long long>(distance) << 32) |
        static_cast<unsigned long long>(hub);
}

__device__ __forceinline__ DeviceQueryResult cuda_unpack_hub_result(
    unsigned long long packed) {

    DeviceQueryResult result;

    if (packed == 0xffffffffffffffffULL) {
        result.distance = INF;
        result.hub = 0xffffffffU;
        return result;
    }

    result.distance =
        static_cast<Weight>(
            static_cast<std::uint32_t>(packed >> 32));
    result.hub = static_cast<Vertex>(packed & 0xffffffffULL);
    return result;
}

__global__ void hub_source_relax_kernel(
    const HubSourceWork* source_work,
    std::uint64_t source_work_count,
    const TargetHubOccurrence* target_occurrences,
    std::uint64_t target_count,
    unsigned long long* packed_results) {

    const std::uint64_t global_thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    const std::uint64_t thread_stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;

    for (std::uint64_t work_index = global_thread;
         work_index < source_work_count;
         work_index += thread_stride) {

        const HubSourceWork work = source_work[work_index];
        const std::uint64_t result_row =
            static_cast<std::uint64_t>(work.source_index) * target_count;

        const Weight left =
            static_cast<Weight>(work.source_distance);

        const Offset target_end =
            work.target_begin + work.target_count;

        for (Offset i = work.target_begin; i < target_end; ++i) {
            const TargetHubOccurrence target =
                target_occurrences[i];

            const Weight candidate64 =
                left + static_cast<Weight>(target.distance);

            const StoredDistance candidate =
                static_cast<StoredDistance>(candidate64);

            const unsigned long long packed =
                cuda_pack_hub_result(candidate, work.hub);

            atomicMin(
                &packed_results[
                    result_row + target.target_index],
                packed);
        }
    }
}

__global__ void hub_source_distance_relax_kernel(
    const HubSourceWork* source_work,
    std::uint64_t source_work_count,
    const TargetHubOccurrence* target_occurrences,
    std::uint64_t target_count,
    StoredDistance* distance_results) {

    const std::uint64_t global_thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    const std::uint64_t thread_stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;

    for (std::uint64_t work_index = global_thread;
         work_index < source_work_count;
         work_index += thread_stride) {

        const HubSourceWork work = source_work[work_index];
        const std::uint64_t result_row =
            static_cast<std::uint64_t>(work.source_index) * target_count;

        const Weight left =
            static_cast<Weight>(work.source_distance);

        const Offset target_end =
            work.target_begin + work.target_count;

        for (Offset i = work.target_begin; i < target_end; ++i) {
            const TargetHubOccurrence target =
                target_occurrences[i];

            const StoredDistance candidate =
                static_cast<StoredDistance>(
                    left + static_cast<Weight>(target.distance));

            atomicMin(
                &distance_results[
                    result_row + target.target_index],
                candidate);
        }
    }
}

__global__ void hub_triple_packed_relax_kernel(
    const HubSourceWork* source_work,
    std::uint64_t source_work_count,
    const TargetHubOccurrence* target_occurrences,
    std::uint64_t target_count,
    unsigned long long* packed_results) {

    for (std::uint64_t work_index = blockIdx.x;
         work_index < source_work_count;
         work_index += gridDim.x) {

        const HubSourceWork work = source_work[work_index];
        const std::uint64_t result_row =
            static_cast<std::uint64_t>(work.source_index) * target_count;

        for (std::uint32_t local_target = threadIdx.x;
             local_target < work.target_count;
             local_target += blockDim.x) {

            const TargetHubOccurrence target =
                target_occurrences[
                    work.target_begin + local_target];

            const Weight candidate64 =
                static_cast<Weight>(work.source_distance) +
                static_cast<Weight>(target.distance);

            const StoredDistance candidate =
                static_cast<StoredDistance>(candidate64);

            const unsigned long long packed =
                cuda_pack_hub_result(candidate, work.hub);

            atomicMin(
                &packed_results[result_row + target.target_index],
                packed);
        }
    }
}

__global__ void hub_triple_distance_relax_kernel(
    const HubSourceWork* source_work,
    std::uint64_t source_work_count,
    const TargetHubOccurrence* target_occurrences,
    std::uint64_t target_count,
    StoredDistance* distance_results) {

    for (std::uint64_t work_index = blockIdx.x;
         work_index < source_work_count;
         work_index += gridDim.x) {

        const HubSourceWork work = source_work[work_index];
        const std::uint64_t result_row =
            static_cast<std::uint64_t>(work.source_index) * target_count;

        for (std::uint32_t local_target = threadIdx.x;
             local_target < work.target_count;
             local_target += blockDim.x) {

            const TargetHubOccurrence target =
                target_occurrences[
                    work.target_begin + local_target];

            const StoredDistance candidate =
                static_cast<StoredDistance>(
                    static_cast<Weight>(work.source_distance) +
                    static_cast<Weight>(target.distance));

            atomicMin(
                &distance_results[result_row + target.target_index],
                candidate);
        }
    }
}

__global__ void hub_packed_finalize_kernel(
    const unsigned long long* packed_results,
    const Vertex* source_ids,
    const Vertex* target_ids,
    std::uint64_t target_count,
    std::uint64_t pair_count,
    DeviceQueryResult* results,
    unsigned long long* block_checksums,
    unsigned long long* block_unreachable) {

    extern __shared__ unsigned long long shared[];
    unsigned long long* shared_checksum = shared;
    unsigned long long* shared_unreachable =
        shared + blockDim.x;

    const std::uint64_t global_thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    const std::uint64_t thread_stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;

    unsigned long long local_checksum = 0;
    unsigned long long local_unreachable = 0;

    for (std::uint64_t pair_index = global_thread;
         pair_index < pair_count;
         pair_index += thread_stride) {

        const std::uint64_t source_index =
            pair_index / target_count;
        const std::uint64_t target_index =
            pair_index - source_index * target_count;

        const DeviceQueryResult result =
            cuda_unpack_hub_result(packed_results[pair_index]);

        if (results != nullptr) {
            results[pair_index] = result;
        }

        local_checksum ^=
            results == nullptr
                ? cuda_distance_fingerprint(
                    source_ids[source_index],
                    target_ids[target_index],
                    result.distance)
                : cuda_query_fingerprint(
                    source_ids[source_index],
                    target_ids[target_index],
                    result.distance,
                    result.hub);

        if (result.distance == INF) {
            ++local_unreachable;
        }
    }

    shared_checksum[threadIdx.x] = local_checksum;
    shared_unreachable[threadIdx.x] = local_unreachable;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2;
         stride != 0;
         stride >>= 1) {

        if (threadIdx.x < stride) {
            shared_checksum[threadIdx.x] ^=
                shared_checksum[threadIdx.x + stride];

            shared_unreachable[threadIdx.x] +=
                shared_unreachable[threadIdx.x + stride];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_checksums[blockIdx.x] = shared_checksum[0];
        block_unreachable[blockIdx.x] = shared_unreachable[0];
    }
}

__global__ void hub_distance_finalize_kernel(
    const StoredDistance* distance_results,
    const Vertex* source_ids,
    const Vertex* target_ids,
    std::uint64_t target_count,
    std::uint64_t pair_count,
    unsigned long long* block_checksums,
    unsigned long long* block_unreachable) {

    extern __shared__ unsigned long long shared[];
    unsigned long long* shared_checksum = shared;
    unsigned long long* shared_unreachable =
        shared + blockDim.x;

    const std::uint64_t global_thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    const std::uint64_t thread_stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;

    unsigned long long local_checksum = 0;
    unsigned long long local_unreachable = 0;

    for (std::uint64_t pair_index = global_thread;
         pair_index < pair_count;
         pair_index += thread_stride) {

        const std::uint64_t source_index =
            pair_index / target_count;
        const std::uint64_t target_index =
            pair_index - source_index * target_count;

        const StoredDistance stored = distance_results[pair_index];
        const Weight distance =
            stored == 0xffffffffU
                ? INF
                : static_cast<Weight>(stored);

        local_checksum ^=
            cuda_distance_fingerprint(
                source_ids[source_index],
                target_ids[target_index],
                distance);

        if (distance == INF) {
            ++local_unreachable;
        }
    }

    shared_checksum[threadIdx.x] = local_checksum;
    shared_unreachable[threadIdx.x] = local_unreachable;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2;
         stride != 0;
         stride >>= 1) {

        if (threadIdx.x < stride) {
            shared_checksum[threadIdx.x] ^=
                shared_checksum[threadIdx.x + stride];
            shared_unreachable[threadIdx.x] +=
                shared_unreachable[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_checksums[blockIdx.x] = shared_checksum[0];
        block_unreachable[blockIdx.x] = shared_unreachable[0];
    }
}

static std::uint64_t checked_cuda_bytes(
    std::size_t count,
    std::size_t element_size,
    const char* description) {

    if (element_size != 0 &&
        count > std::numeric_limits<std::uint64_t>::max() / element_size) {
        throw std::runtime_error(
            std::string("CUDA byte count overflow for ") + description);
    }

    return static_cast<std::uint64_t>(count) * element_size;
}

struct CudaQueryBenchmarkStats {
    std::string kernel;
    std::uint64_t source_count = 0;
    std::uint64_t target_count = 0;
    std::uint64_t pair_count = 0;

    std::uint64_t selected_label_estimated_bytes = 0;
    std::uint64_t host_query_structure_bytes = 0;
    std::uint64_t persistent_workspace_bytes = 0;
    std::uint64_t device_required_bytes = 0;
    std::uint64_t device_free_before_bytes = 0;

    std::uint64_t target_bucket_count = 0;
    std::uint64_t common_hubs = 0;
    std::uint64_t source_work_items = 0;
    std::uint64_t candidate_relaxations = 0;
    std::uint64_t maximum_target_fanout = 0;

    std::uint64_t cpu_threads = 0;
    double cpu_query_seconds = 0.0;

    double resident_selection_seconds = 0.0;
    double preprocess_seconds = 0.0;
    double h2d_allocation_seconds = 0.0;
    double gpu_init_seconds = 0.0;
    double gpu_relax_seconds = 0.0;
    double gpu_finalize_seconds = 0.0;
    double gpu_total_seconds = 0.0;
    double d2h_seconds = 0.0;
    double total_post_resident_seconds = 0.0;

    std::uint64_t unreachable_count = 0;
    std::uint64_t checksum = 0;
};

static std::uint64_t flat_label_capacity_bytes(
    const FlatLabelSet& labels) {

    std::uint64_t total = 0;
    add_host_vector_capacity_bytes(total, labels.offset, "flat label offsets");
    add_host_vector_capacity_bytes(total, labels.hub, "flat label hubs");
    add_host_vector_capacity_bytes(total, labels.distance, "flat label distances");
    return total;
}


static QueryResult intersect_flat_labels_cpu(
    const FlatLabelSet& forward,
    std::uint64_t source_index,
    const FlatLabelSet& reverse,
    std::uint64_t target_index) {

    Offset i = forward.offset[static_cast<std::size_t>(source_index)];
    const Offset i_end =
        forward.offset[static_cast<std::size_t>(source_index) + 1];

    Offset j = reverse.offset[static_cast<std::size_t>(target_index)];
    const Offset j_end =
        reverse.offset[static_cast<std::size_t>(target_index) + 1];

    QueryResult best;

    while (i < i_end && j < j_end) {
        const Vertex a = forward.hub[static_cast<std::size_t>(i)];
        const Vertex b = reverse.hub[static_cast<std::size_t>(j)];

        if (a < b) {
            ++i;
            continue;
        }

        if (b < a) {
            ++j;
            continue;
        }

        const Weight left =
            static_cast<Weight>(
                forward.distance[static_cast<std::size_t>(i)]);
        const Weight right =
            static_cast<Weight>(
                reverse.distance[static_cast<std::size_t>(j)]);

        if (left <= INF - right) {
            const Weight candidate = left + right;
            if (candidate < best.distance) {
                best.distance = candidate;
                best.hub = a;
            }
        }

        ++i;
        ++j;
    }

    return best;
}

static CudaQueryBenchmarkStats run_many_to_many_queries_cpu_pair_staged(
    const std::vector<Vertex>& sources,
    const std::vector<Vertex>& targets,
    const FlatLabelSet& source_labels,
    const FlatLabelSet& target_labels,
    int requested_threads) {

    if (sources.empty() || targets.empty()) {
        throw std::runtime_error(
            "CPU benchmark source and target sets must be non-empty");
    }
    if (requested_threads <= 0) {
        throw std::runtime_error(
            "CPU benchmark thread count must be positive");
    }

#ifndef _OPENMP
    if (requested_threads != 1) {
        throw std::runtime_error(
            "CPU benchmark requested multiple threads, but this executable "
            "was built without OpenMP support");
    }
#endif

    if (source_labels.offset.size() != sources.size() + 1 ||
        target_labels.offset.size() != targets.size() + 1) {
        throw std::runtime_error(
            "CPU benchmark staged-label offsets do not match query sizes");
    }

    if (sources.size() >
        std::numeric_limits<std::uint64_t>::max() / targets.size()) {
        throw std::runtime_error(
            "CPU benchmark pair count overflows uint64_t");
    }

    const std::uint64_t pair_count =
        static_cast<std::uint64_t>(sources.size()) *
        static_cast<std::uint64_t>(targets.size());

    int actual_threads = 1;
#ifdef _OPENMP
#pragma omp parallel num_threads(requested_threads)
    {
#pragma omp single
        {
            actual_threads = omp_get_num_threads();
        }
    }
#endif

    std::vector<std::uint64_t> worker_checksums(
        static_cast<std::size_t>(actual_threads), 0);
    std::vector<std::uint64_t> worker_unreachable(
        static_cast<std::size_t>(actual_threads), 0);

    const std::uint64_t target_count =
        static_cast<std::uint64_t>(targets.size());

    const auto query_start = std::chrono::steady_clock::now();

#ifdef _OPENMP
#pragma omp parallel num_threads(requested_threads)
    {
        const int thread_id = omp_get_thread_num();
        const int thread_count = omp_get_num_threads();

        const std::uint64_t tid =
            static_cast<std::uint64_t>(thread_id);
        const std::uint64_t nthreads =
            static_cast<std::uint64_t>(thread_count);

        const std::uint64_t base = pair_count / nthreads;
        const std::uint64_t remainder = pair_count % nthreads;

        const std::uint64_t begin =
            tid * base + std::min(tid, remainder);
        const std::uint64_t local_count =
            base + (tid < remainder ? 1ULL : 0ULL);
        const std::uint64_t end = begin + local_count;

        std::uint64_t local_checksum = 0;
        std::uint64_t local_unreachable = 0;

        if (begin < end) {
            std::uint64_t source_index = begin / target_count;
            std::uint64_t target_index = begin % target_count;

            for (std::uint64_t pair_index = begin;
                 pair_index < end;
                 ++pair_index) {

                const QueryResult result =
                    intersect_flat_labels_cpu(
                        source_labels,
                        source_index,
                        target_labels,
                        target_index);

                local_checksum ^=
                    distance_result_fingerprint(
                        sources[static_cast<std::size_t>(source_index)],
                        targets[static_cast<std::size_t>(target_index)],
                        result.distance);

                if (result.distance == INF) {
                    ++local_unreachable;
                }

                ++target_index;
                if (target_index == target_count) {
                    target_index = 0;
                    ++source_index;
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

        for (std::uint64_t source_index = 0;
             source_index < static_cast<std::uint64_t>(sources.size());
             ++source_index) {
            for (std::uint64_t target_index = 0;
                 target_index < target_count;
                 ++target_index) {

                const QueryResult result =
                    intersect_flat_labels_cpu(
                        source_labels,
                        source_index,
                        target_labels,
                        target_index);

                local_checksum ^=
                    distance_result_fingerprint(
                        sources[static_cast<std::size_t>(source_index)],
                        targets[static_cast<std::size_t>(target_index)],
                        result.distance);

                if (result.distance == INF) {
                    ++local_unreachable;
                }
            }
        }

        worker_checksums[0] = local_checksum;
        worker_unreachable[0] = local_unreachable;
    }
#endif

    const auto query_end = std::chrono::steady_clock::now();

    CudaQueryBenchmarkStats stats;
    stats.kernel = "cpu-pair";
    stats.source_count = static_cast<std::uint64_t>(sources.size());
    stats.target_count = static_cast<std::uint64_t>(targets.size());
    stats.pair_count = pair_count;
    stats.cpu_threads = static_cast<std::uint64_t>(actual_threads);
    stats.cpu_query_seconds =
        std::chrono::duration<double>(query_end - query_start).count();
    stats.total_post_resident_seconds = stats.cpu_query_seconds;

    stats.selected_label_estimated_bytes =
        flat_label_capacity_bytes(source_labels) +
        flat_label_capacity_bytes(target_labels);

    stats.host_query_structure_bytes =
        static_cast<std::uint64_t>(actual_threads) *
        2ULL * sizeof(std::uint64_t);

    for (std::size_t i = 0; i < worker_checksums.size(); ++i) {
        stats.checksum ^= worker_checksums[i];
        stats.unreachable_count += worker_unreachable[i];
    }

    return stats;
}

static CudaQueryBenchmarkStats run_many_to_many_queries_cuda(
    DiskHubLabels& labels,
    const ResidentHubLabels* resident_labels,
    const std::vector<Vertex>& sources,
    const std::vector<Vertex>& targets,
    std::uint64_t cache_limit_bytes,
    bool quiet,
    int device_id,
    bool print_report = true,
    const FlatLabelSet* staged_source_labels = nullptr,
    const FlatLabelSet* staged_target_labels = nullptr) {

    if (cache_limit_bytes == 0) {
        throw std::runtime_error(
            "Query cache size must be greater than zero");
    }

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

    CudaQueryBenchmarkStats stats;
    stats.kernel = "pair";
    stats.source_count = static_cast<std::uint64_t>(sources.size());
    stats.target_count = static_cast<std::uint64_t>(targets.size());
    stats.pair_count = pair_count;

    const std::uint64_t source_bytes =
        resident_labels != nullptr
            ? estimate_selected_label_bytes(
                resident_labels->forward, sources)
            : estimate_selected_label_bytes(
                labels.forward, sources);
    const std::uint64_t target_bytes =
        resident_labels != nullptr
            ? estimate_selected_label_bytes(
                resident_labels->reverse, targets)
            : estimate_selected_label_bytes(
                labels.reverse, targets);

    if (source_bytes == std::numeric_limits<std::uint64_t>::max() ||
        target_bytes == std::numeric_limits<std::uint64_t>::max() ||
        source_bytes >
            std::numeric_limits<std::uint64_t>::max() - target_bytes) {
        throw std::runtime_error(
            "Estimated memory required by selected labels overflows uint64_t");
    }

    const std::uint64_t total_label_bytes =
        source_bytes + target_bytes;

    stats.selected_label_estimated_bytes = total_label_bytes;

    if (total_label_bytes > cache_limit_bytes) {
        std::ostringstream message;
        message
            << "All required labels need approximately "
            << std::fixed << std::setprecision(2)
            << (total_label_bytes / (1024.0 * 1024.0))
            << " MiB, exceeding --cache-mb "
            << (cache_limit_bytes / (1024ULL * 1024ULL))
            << ". Increase --cache-mb so every required label can be "
               "preloaded before timing.";
        throw std::runtime_error(message.str());
    }

    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");

    if (device_id < 0 || device_id >= device_count) {
        throw std::runtime_error(
            "CUDA device ID is out of range; available devices: " +
            std::to_string(device_count));
    }

    cuda_check(cudaSetDevice(device_id), "cudaSetDevice");

    cudaDeviceProp properties{};
    cuda_check(
        cudaGetDeviceProperties(&properties, device_id),
        "cudaGetDeviceProperties");

    cuda_check(cudaFree(nullptr), "CUDA runtime initialization");

    if (print_report) {
    std::cerr
        << "many-to-many CUDA query: "
        << sources.size() << " sources x "
        << targets.size() << " targets = "
        << pair_count << " pairs\n"
        << "CUDA device " << device_id << ": "
        << properties.name << "\n"
        << "estimated host preloaded label memory: "
        << std::fixed << std::setprecision(2)
        << (total_label_bytes / (1024.0 * 1024.0))
        << " MiB, limit: "
        << (cache_limit_bytes / (1024.0 * 1024.0))
        << " MiB\n";
    }

    if ((staged_source_labels == nullptr) !=
        (staged_target_labels == nullptr)) {
        throw std::runtime_error(
            "Internal error: staged source/target labels must be supplied together");
    }
    if (resident_labels != nullptr && staged_source_labels != nullptr) {
        throw std::runtime_error(
            "Internal error: resident HL and benchmark-staged labels are mutually exclusive");
    }

    FlatLabelSet owned_source_labels;
    FlatLabelSet owned_target_labels;
    const FlatLabelSet* source_labels_ptr = staged_source_labels;
    const FlatLabelSet* target_labels_ptr = staged_target_labels;
    double preload_seconds = 0.0;
    double resident_stage_seconds = 0.0;

    if (source_labels_ptr == nullptr && resident_labels != nullptr) {
        const auto stage_start =
            std::chrono::steady_clock::now();

        owned_source_labels =
            preload_flat_label_set_resident(
                resident_labels->forward,
                sources,
                resident_labels->num_vertices,
                "forward label");
        owned_target_labels =
            preload_flat_label_set_resident(
                resident_labels->reverse,
                targets,
                resident_labels->num_vertices,
                "reverse label");

        const auto stage_end =
            std::chrono::steady_clock::now();
        resident_stage_seconds =
            std::chrono::duration<double>(
                stage_end - stage_start).count();
        source_labels_ptr = &owned_source_labels;
        target_labels_ptr = &owned_target_labels;
    } else if (source_labels_ptr == nullptr) {
        const auto preload_start =
            std::chrono::steady_clock::now();

        owned_source_labels =
            preload_flat_label_set(
                labels.input,
                labels.forward,
                sources,
                labels.num_vertices,
                "forward label");
        owned_target_labels =
            preload_flat_label_set(
                labels.input,
                labels.reverse,
                targets,
                labels.num_vertices,
                "reverse label");

        const auto preload_end =
            std::chrono::steady_clock::now();
        preload_seconds =
            std::chrono::duration<double>(
                preload_end - preload_start).count();
        source_labels_ptr = &owned_source_labels;
        target_labels_ptr = &owned_target_labels;
    }

    const FlatLabelSet& source_labels = *source_labels_ptr;
    const FlatLabelSet& target_labels = *target_labels_ptr;

    stats.resident_selection_seconds = resident_stage_seconds;
    stats.host_query_structure_bytes =
        staged_source_labels != nullptr
            ? 0
            : flat_label_capacity_bytes(source_labels) +
                flat_label_capacity_bytes(target_labels);

    constexpr unsigned int threads_per_block = 256;

    const std::uint64_t block_count64 =
        (pair_count + threads_per_block - 1) /
        threads_per_block;

    if (block_count64 >
        static_cast<std::uint64_t>(properties.maxGridSize[0])) {
        throw std::runtime_error(
            "Query has too many pairs for the current one-thread-per-pair "
            "CUDA launch grid");
    }

    const std::size_t block_count =
        static_cast<std::size_t>(block_count64);

    std::uint64_t required_device_bytes = 0;

    auto add_device_bytes = [&](std::uint64_t bytes) {
        if (required_device_bytes >
            std::numeric_limits<std::uint64_t>::max() - bytes) {
            throw std::runtime_error(
                "Required CUDA memory byte count overflows uint64_t");
        }
        required_device_bytes += bytes;
    };

    add_device_bytes(checked_cuda_bytes(
        source_labels.offset.size(), sizeof(Offset), "source offsets"));
    add_device_bytes(checked_cuda_bytes(
        source_labels.hub.size(), sizeof(Vertex), "source hubs"));
    add_device_bytes(checked_cuda_bytes(
        source_labels.distance.size(), sizeof(StoredDistance),
        "source distances"));
    add_device_bytes(checked_cuda_bytes(
        target_labels.offset.size(), sizeof(Offset), "target offsets"));
    add_device_bytes(checked_cuda_bytes(
        target_labels.hub.size(), sizeof(Vertex), "target hubs"));
    add_device_bytes(checked_cuda_bytes(
        target_labels.distance.size(), sizeof(StoredDistance),
        "target distances"));
    add_device_bytes(checked_cuda_bytes(
        sources.size(), sizeof(Vertex), "source IDs"));
    add_device_bytes(checked_cuda_bytes(
        targets.size(), sizeof(Vertex), "target IDs"));
    add_device_bytes(checked_cuda_bytes(
        block_count, sizeof(unsigned long long), "block checksums"));
    add_device_bytes(checked_cuda_bytes(
        block_count, sizeof(unsigned long long), "block unreachable counts"));

    if (!quiet) {
        if (pair_count >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error(
                "Too many CUDA query results to store; use --quiet");
        }

        add_device_bytes(checked_cuda_bytes(
            static_cast<std::size_t>(pair_count),
            sizeof(DeviceQueryResult),
            "query results"));
    }

    stats.device_required_bytes = required_device_bytes;

    std::size_t free_device_bytes = 0;
    std::size_t total_device_bytes = 0;
    cuda_check(
        cudaMemGetInfo(&free_device_bytes, &total_device_bytes),
        "cudaMemGetInfo");
    stats.device_free_before_bytes =
        static_cast<std::uint64_t>(free_device_bytes);

    if (print_report) {
    std::cerr
        << "estimated CUDA memory required: "
        << (required_device_bytes / (1024.0 * 1024.0))
        << " MiB; currently free: "
        << (free_device_bytes / (1024.0 * 1024.0))
        << " MiB\n";
    }

    if (required_device_bytes >
        static_cast<std::uint64_t>(free_device_bytes)) {
        throw std::runtime_error(
            "Insufficient free CUDA device memory for selected labels/results");
    }

    const auto transfer_start =
        std::chrono::steady_clock::now();

    CudaBuffer<Offset> d_source_offset(source_labels.offset.size());
    CudaBuffer<Vertex> d_source_hub(source_labels.hub.size());
    CudaBuffer<StoredDistance> d_source_distance(
        source_labels.distance.size());

    CudaBuffer<Offset> d_target_offset(target_labels.offset.size());
    CudaBuffer<Vertex> d_target_hub(target_labels.hub.size());
    CudaBuffer<StoredDistance> d_target_distance(
        target_labels.distance.size());

    CudaBuffer<Vertex> d_sources(sources.size());
    CudaBuffer<Vertex> d_targets(targets.size());

    CudaBuffer<unsigned long long> d_block_checksums(block_count);
    CudaBuffer<unsigned long long> d_block_unreachable(block_count);

    CudaBuffer<DeviceQueryResult> d_results;
    if (!quiet) {
        d_results.allocate(static_cast<std::size_t>(pair_count));
    }

    d_source_offset.upload(source_labels.offset);
    d_source_hub.upload(source_labels.hub);
    d_source_distance.upload(source_labels.distance);
    d_target_offset.upload(target_labels.offset);
    d_target_hub.upload(target_labels.hub);
    d_target_distance.upload(target_labels.distance);
    d_sources.upload(sources);
    d_targets.upload(targets);

    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize before timing");

    const auto transfer_end =
        std::chrono::steady_clock::now();

    const double transfer_seconds =
        std::chrono::duration<double>(
            transfer_end - transfer_start).count();

    stats.h2d_allocation_seconds = transfer_seconds;

    cudaEvent_t start_event = nullptr;
    cudaEvent_t end_event = nullptr;

    cuda_check(cudaEventCreate(&start_event), "cudaEventCreate start");

    try {
        cuda_check(cudaEventCreate(&end_event), "cudaEventCreate end");

        const std::size_t shared_bytes =
            2ULL * threads_per_block * sizeof(unsigned long long);

        cuda_check(cudaEventRecord(start_event), "cudaEventRecord start");

        hub_label_query_kernel<<<
            static_cast<unsigned int>(block_count),
            threads_per_block,
            shared_bytes>>>(
                d_source_offset.data(),
                d_source_hub.data(),
                d_source_distance.data(),
                d_target_offset.data(),
                d_target_hub.data(),
                d_target_distance.data(),
                d_sources.data(),
                d_targets.data(),
                static_cast<std::uint64_t>(sources.size()),
                static_cast<std::uint64_t>(targets.size()),
                pair_count,
                quiet ? nullptr : d_results.data(),
                d_block_checksums.data(),
                d_block_unreachable.data());

        cuda_check(cudaGetLastError(), "CUDA query kernel launch");
        cuda_check(cudaEventRecord(end_event), "cudaEventRecord end");
        cuda_check(cudaEventSynchronize(end_event), "cudaEventSynchronize end");

        float query_milliseconds = 0.0f;
        cuda_check(
            cudaEventElapsedTime(
                &query_milliseconds,
                start_event,
                end_event),
            "cudaEventElapsedTime");

        cudaEventDestroy(start_event);
        start_event = nullptr;
        cudaEventDestroy(end_event);
        end_event = nullptr;

        const auto download_start =
            std::chrono::steady_clock::now();

        std::vector<unsigned long long> block_checksums;
        std::vector<unsigned long long> block_unreachable;
        std::vector<DeviceQueryResult> device_results;

        d_block_checksums.download(block_checksums);
        d_block_unreachable.download(block_unreachable);
        if (!quiet) {
            d_results.download(device_results);
        }

        const auto download_end =
            std::chrono::steady_clock::now();
        const double download_seconds =
            std::chrono::duration<double>(
                download_end - download_start).count();

        std::uint64_t checksum = 0;
        std::uint64_t unreachable_count = 0;

        for (std::size_t i = 0; i < block_checksums.size(); ++i) {
            checksum ^= static_cast<std::uint64_t>(block_checksums[i]);
            unreachable_count +=
                static_cast<std::uint64_t>(block_unreachable[i]);
        }

        const double query_seconds =
            static_cast<double>(query_milliseconds) / 1000.0;

        const double average_ns =
            pair_count == 0
                ? 0.0
                : (static_cast<double>(query_milliseconds) * 1.0e6) /
                    static_cast<double>(pair_count);

        const double throughput =
            query_seconds > 0.0
                ? static_cast<double>(pair_count) / query_seconds
                : 0.0;

        const double post_preload_seconds =
            resident_stage_seconds +
            transfer_seconds + query_seconds + download_seconds;
        const double post_preload_ns =
            post_preload_seconds * 1.0e9 /
            static_cast<double>(pair_count);
        const double post_preload_throughput =
            post_preload_seconds > 0.0
                ? static_cast<double>(pair_count) / post_preload_seconds
                : 0.0;

        stats.gpu_relax_seconds = query_seconds;
        stats.gpu_total_seconds = query_seconds;
        stats.d2h_seconds = download_seconds;
        stats.total_post_resident_seconds = post_preload_seconds;
        stats.unreachable_count = unreachable_count;
        stats.checksum = checksum;

        if (print_report) {
        std::cerr
            << (resident_labels != nullptr
                    ? "query-time resident-label selection/flattening: "
                    : "labels staged from .hlbin (excluded; models resident HL): ")
            << std::fixed << std::setprecision(6)
            << (resident_labels != nullptr
                    ? resident_stage_seconds
                    : preload_seconds)
            << " s\n"
            << "H2D + CUDA allocation time: "
            << transfer_seconds << " s\n"
            << "CUDA kernel time: "
            << query_seconds << " s\n"
            << "D2H result/validation time: "
            << download_seconds << " s\n"
            << "TOTAL post-resident-label query time: "
            << post_preload_seconds << " s\n"
            << "average CUDA kernel time: "
            << std::setprecision(2)
            << average_ns << " ns/pair\n"
            << "average TOTAL post-resident-label time: "
            << post_preload_ns << " ns/pair\n"
            << "kernel throughput: "
            << throughput << " pairs/s\n"
            << "end-to-end post-resident-label throughput: "
            << post_preload_throughput << " pairs/s\n"
            << "checksum mode: "
            << (quiet ? "distance-only" : "distance+hub") << "\n"
            << "query pairs completed: "
            << pair_count
            << ", unreachable: "
            << unreachable_count
            << ", checksum: "
            << checksum << "\n";
        }

        if (!quiet) {
            std::size_t result_index = 0;

            for (Vertex source : sources) {
                for (Vertex target : targets) {
                    QueryResult result;
                    result.distance =
                        device_results[result_index].distance;
                    result.hub =
                        device_results[result_index].hub;

                    print_query_result(source, target, result);
                    ++result_index;
                }
            }
        }
    } catch (...) {
        if (start_event != nullptr) {
            cudaEventDestroy(start_event);
        }
        if (end_event != nullptr) {
            cudaEventDestroy(end_event);
        }
        throw;
    }

    return stats;
}


static CudaQueryBenchmarkStats run_many_to_many_queries_cuda_hub(
    DiskHubLabels& labels,
    const ResidentHubLabels* resident_labels,
    const std::vector<Vertex>& sources,
    const std::vector<Vertex>& targets,
    std::uint64_t cache_limit_bytes,
    bool quiet,
    int device_id,
    const std::string& kernel_mode,
    bool print_report = true,
    HubBucketWorkspace* reusable_bucket_workspace = nullptr,
    const FlatLabelSet* staged_source_labels = nullptr,
    const FlatLabelSet* staged_target_labels = nullptr) {

    const bool hub_source_mode = kernel_mode == "hub-source";
    const bool hub_triple_mode = kernel_mode == "hub-triple";

    if (!hub_source_mode && !hub_triple_mode) {
        throw std::runtime_error(
            "Internal error: invalid hub-centric CUDA kernel mode");
    }

    if (cache_limit_bytes == 0) {
        throw std::runtime_error(
            "Query cache size must be greater than zero");
    }

    if (sources.empty() || targets.empty()) {
        throw std::runtime_error(
            "Source and target sets must both be non-empty");
    }

    if (sources.size() > std::numeric_limits<std::uint32_t>::max() ||
        targets.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "Hub-centric CUDA requires source and target counts to fit in uint32_t");
    }

    if (sources.size() >
        std::numeric_limits<std::uint64_t>::max() / targets.size()) {
        throw std::runtime_error(
            "Number of source-target pairs overflows uint64_t");
    }

    const std::uint64_t pair_count =
        static_cast<std::uint64_t>(sources.size()) *
        static_cast<std::uint64_t>(targets.size());

    CudaQueryBenchmarkStats stats;
    stats.kernel = kernel_mode;
    stats.source_count = static_cast<std::uint64_t>(sources.size());
    stats.target_count = static_cast<std::uint64_t>(targets.size());
    stats.pair_count = pair_count;

    if (pair_count >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(
            "Hub-centric CUDA result matrix does not fit in size_t");
    }

    const std::uint64_t source_bytes =
        resident_labels != nullptr
            ? estimate_selected_label_bytes(
                resident_labels->forward, sources)
            : estimate_selected_label_bytes(
                labels.forward, sources);
    const std::uint64_t target_bytes =
        resident_labels != nullptr
            ? estimate_selected_label_bytes(
                resident_labels->reverse, targets)
            : estimate_selected_label_bytes(
                labels.reverse, targets);

    if (source_bytes == std::numeric_limits<std::uint64_t>::max() ||
        target_bytes == std::numeric_limits<std::uint64_t>::max() ||
        source_bytes >
            std::numeric_limits<std::uint64_t>::max() - target_bytes) {
        throw std::runtime_error(
            "Estimated memory required by selected labels overflows uint64_t");
    }

    const std::uint64_t total_label_bytes =
        source_bytes + target_bytes;

    stats.selected_label_estimated_bytes = total_label_bytes;

    if (resident_labels == nullptr &&
        total_label_bytes > cache_limit_bytes) {
        std::ostringstream message;
        message
            << "All required labels need approximately "
            << std::fixed << std::setprecision(2)
            << (total_label_bytes / (1024.0 * 1024.0))
            << " MiB, exceeding --cache-mb "
            << (cache_limit_bytes / (1024ULL * 1024ULL))
            << ". Increase --cache-mb so every required label can be "
               "preloaded before timing.";
        throw std::runtime_error(message.str());
    }

    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");

    if (device_id < 0 || device_id >= device_count) {
        throw std::runtime_error(
            "CUDA device ID is out of range; available devices: " +
            std::to_string(device_count));
    }

    cuda_check(cudaSetDevice(device_id), "cudaSetDevice");

    cudaDeviceProp properties{};
    cuda_check(
        cudaGetDeviceProperties(&properties, device_id),
        "cudaGetDeviceProperties");

    cuda_check(cudaFree(nullptr), "CUDA runtime initialization");

    if (print_report) {
    std::cerr
        << "many-to-many CUDA query: "
        << sources.size() << " sources x "
        << targets.size() << " targets = "
        << pair_count << " pairs\n"
        << "CUDA device " << device_id << ": "
        << properties.name << "\n"
        << "CUDA query kernel: " << kernel_mode << "\n"
        << "selected-label footprint: "
        << std::fixed << std::setprecision(2)
        << (total_label_bytes / (1024.0 * 1024.0))
        << " MiB"
        << (resident_labels != nullptr
                ? " (direct resident scan)\n"
                : " (disk-staged before query timing)\n");
    }

    if ((staged_source_labels == nullptr) !=
        (staged_target_labels == nullptr)) {
        throw std::runtime_error(
            "Internal error: staged source/target labels must be supplied together");
    }
    if (resident_labels != nullptr && staged_source_labels != nullptr) {
        throw std::runtime_error(
            "Internal error: resident HL and benchmark-staged labels are mutually exclusive");
    }

    FlatLabelSet owned_source_labels;
    FlatLabelSet owned_target_labels;
    const FlatLabelSet* source_labels_ptr = staged_source_labels;
    const FlatLabelSet* target_labels_ptr = staged_target_labels;
    double preload_seconds = 0.0;
    double resident_stage_seconds = 0.0;

    if (resident_labels == nullptr && source_labels_ptr == nullptr) {

        const auto preload_start =
            std::chrono::steady_clock::now();

        owned_source_labels =
            preload_flat_label_set(
                labels.input,
                labels.forward,
                sources,
                labels.num_vertices,
                "forward label");
        owned_target_labels =
            preload_flat_label_set(
                labels.input,
                labels.reverse,
                targets,
                labels.num_vertices,
                "reverse label");

        const auto preload_end =
            std::chrono::steady_clock::now();
        preload_seconds =
            std::chrono::duration<double>(
                preload_end - preload_start).count();
        source_labels_ptr = &owned_source_labels;
        target_labels_ptr = &owned_target_labels;
    }

    std::unique_ptr<HubBucketWorkspace> owned_bucket_workspace;
    HubBucketWorkspace* bucket_workspace_ptr = reusable_bucket_workspace;
    if (bucket_workspace_ptr == nullptr) {
        owned_bucket_workspace =
            std::make_unique<HubBucketWorkspace>(labels.num_vertices);
        bucket_workspace_ptr = owned_bucket_workspace.get();
    }
    HubBucketWorkspace& bucket_workspace = *bucket_workspace_ptr;

    stats.persistent_workspace_bytes =
        checked_host_capacity_bytes(
            bucket_workspace.stamp.capacity(),
            sizeof(std::uint32_t),
            "hub stamp workspace") +
        checked_host_capacity_bytes(
            bucket_workspace.bucket_id.capacity(),
            sizeof(std::uint32_t),
            "hub bucket-id workspace");

    constexpr std::uint32_t threads_per_block = 256;

    // query-dependent CPU preprocessing starts here
    const auto preprocess_start =
        std::chrono::steady_clock::now();

    HubInvertedWork inverted =
        resident_labels != nullptr
            ? build_hub_bucket_work_no_sort_resident(
                resident_labels->forward,
                sources,
                resident_labels->reverse,
                targets,
                bucket_workspace)
            : build_hub_bucket_work_no_sort(
                *source_labels_ptr,
                *target_labels_ptr,
                bucket_workspace);

    const auto preprocess_end =
        std::chrono::steady_clock::now();

    const double preprocess_seconds =
        std::chrono::duration<double>(
            preprocess_end - preprocess_start).count();

    stats.preprocess_seconds = preprocess_seconds;
    stats.host_query_structure_bytes = inverted.peak_query_host_bytes;
    stats.target_bucket_count = inverted.target_bucket_count;
    stats.common_hubs = inverted.common_hubs;
    stats.source_work_items =
        static_cast<std::uint64_t>(inverted.source_work.size());
    stats.candidate_relaxations = inverted.candidate_relaxations;
    stats.maximum_target_fanout = inverted.maximum_target_fanout;

    if (inverted.maximum_candidate_distance >=
        static_cast<Weight>(
            std::numeric_limits<StoredDistance>::max())) {
        std::ostringstream message;
        message
            << "Hub-centric CUDA reserves UINT32_MAX as infinity, so every "
               "candidate distance must be smaller than UINT32_MAX. "
               "Preprocessing found a possible candidate of "
            << inverted.maximum_candidate_distance
            << ". Use --cuda-kernel pair for this query set.";
        throw std::runtime_error(message.str());
    }

    const std::uint64_t max_grid_x =
        static_cast<std::uint64_t>(properties.maxGridSize[0]);

    if (max_grid_x == 0) {
        throw std::runtime_error(
            "CUDA device reports a zero grid-x limit");
    }

    const std::uint64_t validation_blocks64 =
        std::min(
            max_grid_x,
            (pair_count + threads_per_block - 1) /
                threads_per_block);

    if (validation_blocks64 == 0) {
        throw std::runtime_error(
            "Internal error: zero validation blocks");
    }

    const std::size_t validation_blocks =
        static_cast<std::size_t>(validation_blocks64);

    std::uint64_t relaxation_blocks64 = 0;

    if (!inverted.source_work.empty()) {
        if (hub_source_mode) {
            // one thread per (hub,source)
            relaxation_blocks64 =
                std::min(
                    max_grid_x,
                    (static_cast<std::uint64_t>(
                        inverted.source_work.size()) +
                     threads_per_block - 1) /
                        threads_per_block);
        } else {
            // one block per (hub,source), threads cooperate on its targets
            relaxation_blocks64 =
                std::min(
                    max_grid_x,
                    static_cast<std::uint64_t>(
                        inverted.source_work.size()));
        }
    }

    const bool distance_only_results = quiet;

    std::uint64_t required_device_bytes = 0;

    auto add_device_bytes = [&](std::uint64_t bytes) {
        if (required_device_bytes >
            std::numeric_limits<std::uint64_t>::max() - bytes) {
            throw std::runtime_error(
                "Required CUDA memory byte count overflows uint64_t");
        }
        required_device_bytes += bytes;
    };

    add_device_bytes(checked_cuda_bytes(
        inverted.target_occurrences.size(),
        sizeof(TargetHubOccurrence),
        "hub target occurrences"));
    add_device_bytes(checked_cuda_bytes(
        inverted.source_work.size(),
        sizeof(HubSourceWork),
        "hub-source work items"));

    add_device_bytes(checked_cuda_bytes(
        static_cast<std::size_t>(pair_count),
        distance_only_results
            ? sizeof(StoredDistance)
            : sizeof(unsigned long long),
        "hub-centric result matrix"));

    add_device_bytes(checked_cuda_bytes(
        sources.size(), sizeof(Vertex), "source IDs"));
    add_device_bytes(checked_cuda_bytes(
        targets.size(), sizeof(Vertex), "target IDs"));
    add_device_bytes(checked_cuda_bytes(
        validation_blocks,
        sizeof(unsigned long long),
        "block checksums"));
    add_device_bytes(checked_cuda_bytes(
        validation_blocks,
        sizeof(unsigned long long),
        "block unreachable counts"));

    if (!quiet) {
        add_device_bytes(checked_cuda_bytes(
            static_cast<std::size_t>(pair_count),
            sizeof(DeviceQueryResult),
            "decoded query results"));
    }

    stats.device_required_bytes = required_device_bytes;

    std::size_t free_device_bytes = 0;
    std::size_t total_device_bytes = 0;
    cuda_check(
        cudaMemGetInfo(&free_device_bytes, &total_device_bytes),
        "cudaMemGetInfo");
    stats.device_free_before_bytes =
        static_cast<std::uint64_t>(free_device_bytes);

    if (required_device_bytes >
        static_cast<std::uint64_t>(free_device_bytes)) {
        throw std::runtime_error(
            "Insufficient free CUDA device memory for hub-centric query");
    }

    // allocation + host-to-device transfer
    const auto transfer_start =
        std::chrono::steady_clock::now();

    CudaBuffer<TargetHubOccurrence> d_target_occurrences(
        inverted.target_occurrences.size());
    CudaBuffer<HubSourceWork> d_source_work(
        inverted.source_work.size());

    CudaBuffer<unsigned long long> d_packed_results;
    CudaBuffer<StoredDistance> d_distance_results;

    if (distance_only_results) {
        d_distance_results.allocate(
            static_cast<std::size_t>(pair_count));
    } else {
        d_packed_results.allocate(
            static_cast<std::size_t>(pair_count));
    }

    CudaBuffer<Vertex> d_sources(sources.size());
    CudaBuffer<Vertex> d_targets(targets.size());

    CudaBuffer<unsigned long long> d_block_checksums(
        validation_blocks);
    CudaBuffer<unsigned long long> d_block_unreachable(
        validation_blocks);

    CudaBuffer<DeviceQueryResult> d_results;
    if (!quiet) {
        d_results.allocate(
            static_cast<std::size_t>(pair_count));
    }

    d_target_occurrences.upload(inverted.target_occurrences);
    d_source_work.upload(inverted.source_work);
    d_sources.upload(sources);
    d_targets.upload(targets);

    cuda_check(
        cudaDeviceSynchronize(),
        "cudaDeviceSynchronize after hub-centric H2D transfer");

    const auto transfer_end =
        std::chrono::steady_clock::now();

    const double transfer_seconds =
        std::chrono::duration<double>(
            transfer_end - transfer_start).count();

    stats.h2d_allocation_seconds = transfer_seconds;

    cudaEvent_t start_event = nullptr;
    cudaEvent_t after_init_event = nullptr;
    cudaEvent_t after_relax_event = nullptr;
    cudaEvent_t end_event = nullptr;

    cuda_check(
        cudaEventCreate(&start_event),
        "cudaEventCreate start");

    std::vector<DeviceQueryResult> downloaded_results;

    try {
        cuda_check(
            cudaEventCreate(&after_init_event),
            "cudaEventCreate after-init");
        cuda_check(
            cudaEventCreate(&after_relax_event),
            "cudaEventCreate after-relax");
        cuda_check(
            cudaEventCreate(&end_event),
            "cudaEventCreate end");

        const std::size_t shared_bytes =
            2ULL * threads_per_block * sizeof(unsigned long long);

        cuda_check(
            cudaEventRecord(start_event),
            "cudaEventRecord start");

        if (distance_only_results) {
            cuda_check(
                cudaMemsetAsync(
                    d_distance_results.data(),
                    0xff,
                    static_cast<std::size_t>(pair_count) *
                        sizeof(StoredDistance)),
                "cudaMemsetAsync distance results");
        } else {
            cuda_check(
                cudaMemsetAsync(
                    d_packed_results.data(),
                    0xff,
                    static_cast<std::size_t>(pair_count) *
                        sizeof(unsigned long long)),
                "cudaMemsetAsync packed results");
        }

        cuda_check(
            cudaEventRecord(after_init_event),
            "cudaEventRecord after-init");

        if (relaxation_blocks64 != 0) {
            if (hub_source_mode && distance_only_results) {
                hub_source_distance_relax_kernel<<<
                    static_cast<unsigned int>(relaxation_blocks64),
                    threads_per_block>>>(
                        d_source_work.data(),
                        static_cast<std::uint64_t>(
                            inverted.source_work.size()),
                        d_target_occurrences.data(),
                        static_cast<std::uint64_t>(targets.size()),
                        d_distance_results.data());
            } else if (hub_source_mode) {
                hub_source_relax_kernel<<<
                    static_cast<unsigned int>(relaxation_blocks64),
                    threads_per_block>>>(
                        d_source_work.data(),
                        static_cast<std::uint64_t>(
                            inverted.source_work.size()),
                        d_target_occurrences.data(),
                        static_cast<std::uint64_t>(targets.size()),
                        d_packed_results.data());
            } else if (distance_only_results) {
                hub_triple_distance_relax_kernel<<<
                    static_cast<unsigned int>(relaxation_blocks64),
                    threads_per_block>>>(
                        d_source_work.data(),
                        static_cast<std::uint64_t>(
                            inverted.source_work.size()),
                        d_target_occurrences.data(),
                        static_cast<std::uint64_t>(targets.size()),
                        d_distance_results.data());
            } else {
                hub_triple_packed_relax_kernel<<<
                    static_cast<unsigned int>(relaxation_blocks64),
                    threads_per_block>>>(
                        d_source_work.data(),
                        static_cast<std::uint64_t>(
                            inverted.source_work.size()),
                        d_target_occurrences.data(),
                        static_cast<std::uint64_t>(targets.size()),
                        d_packed_results.data());
            }

            cuda_check(
                cudaGetLastError(),
                "CUDA hub-centric relaxation kernel launch");
        }

        cuda_check(
            cudaEventRecord(after_relax_event),
            "cudaEventRecord after-relax");

        if (distance_only_results) {
            hub_distance_finalize_kernel<<<
                static_cast<unsigned int>(validation_blocks64),
                threads_per_block,
                shared_bytes>>>(
                    d_distance_results.data(),
                    d_sources.data(),
                    d_targets.data(),
                    static_cast<std::uint64_t>(targets.size()),
                    pair_count,
                    d_block_checksums.data(),
                    d_block_unreachable.data());
        } else {
            hub_packed_finalize_kernel<<<
                static_cast<unsigned int>(validation_blocks64),
                threads_per_block,
                shared_bytes>>>(
                    d_packed_results.data(),
                    d_sources.data(),
                    d_targets.data(),
                    static_cast<std::uint64_t>(targets.size()),
                    pair_count,
                    quiet ? nullptr : d_results.data(),
                    d_block_checksums.data(),
                    d_block_unreachable.data());
        }

        cuda_check(
            cudaGetLastError(),
            "CUDA hub-centric finalize kernel launch");

        cuda_check(
            cudaEventRecord(end_event),
            "cudaEventRecord end");
        cuda_check(
            cudaEventSynchronize(end_event),
            "cudaEventSynchronize end");

        float init_ms = 0.0f;
        float relax_ms = 0.0f;
        float finalize_ms = 0.0f;
        float total_ms = 0.0f;

        cuda_check(
            cudaEventElapsedTime(
                &init_ms,
                start_event,
                after_init_event),
            "cudaEventElapsedTime initialization");
        cuda_check(
            cudaEventElapsedTime(
                &relax_ms,
                after_init_event,
                after_relax_event),
            "cudaEventElapsedTime relaxation");
        cuda_check(
            cudaEventElapsedTime(
                &finalize_ms,
                after_relax_event,
                end_event),
            "cudaEventElapsedTime finalize");
        cuda_check(
            cudaEventElapsedTime(
                &total_ms,
                start_event,
                end_event),
            "cudaEventElapsedTime total");

        cudaEventDestroy(start_event);
        start_event = nullptr;
        cudaEventDestroy(after_init_event);
        after_init_event = nullptr;
        cudaEventDestroy(after_relax_event);
        after_relax_event = nullptr;
        cudaEventDestroy(end_event);
        end_event = nullptr;

        // device-to-host result/validation transfer
        const auto download_start =
            std::chrono::steady_clock::now();

        std::vector<unsigned long long> block_checksums;
        std::vector<unsigned long long> block_unreachable;
        d_block_checksums.download(block_checksums);
        d_block_unreachable.download(block_unreachable);

        if (!quiet) {
            d_results.download(downloaded_results);
        }

        const auto download_end =
            std::chrono::steady_clock::now();

        const double download_seconds =
            std::chrono::duration<double>(
                download_end - download_start).count();

        std::uint64_t checksum = 0;
        std::uint64_t unreachable_count = 0;

        for (std::size_t i = 0; i < block_checksums.size(); ++i) {
            checksum ^=
                static_cast<std::uint64_t>(block_checksums[i]);
            unreachable_count +=
                static_cast<std::uint64_t>(block_unreachable[i]);
        }

        const double gpu_seconds =
            static_cast<double>(total_ms) / 1000.0;
        const double relax_seconds =
            static_cast<double>(relax_ms) / 1000.0;

        const double post_preload_seconds =
            resident_stage_seconds +
            preprocess_seconds +
            transfer_seconds +
            gpu_seconds +
            download_seconds;

        const double average_post_preload_ns =
            post_preload_seconds * 1.0e9 /
            static_cast<double>(pair_count);

        const double end_to_end_pair_throughput =
            post_preload_seconds > 0.0
                ? static_cast<double>(pair_count) /
                    post_preload_seconds
                : 0.0;

        const double candidate_throughput =
            relax_seconds > 0.0
                ? static_cast<double>(
                    inverted.candidate_relaxations) /
                    relax_seconds
                : 0.0;

        stats.resident_selection_seconds = resident_stage_seconds;
        stats.gpu_init_seconds = static_cast<double>(init_ms) / 1000.0;
        stats.gpu_relax_seconds = relax_seconds;
        stats.gpu_finalize_seconds =
            static_cast<double>(finalize_ms) / 1000.0;
        stats.gpu_total_seconds = gpu_seconds;
        stats.d2h_seconds = download_seconds;
        stats.total_post_resident_seconds = post_preload_seconds;
        stats.unreachable_count = unreachable_count;
        stats.checksum = checksum;

        if (print_report) {
        if (resident_labels != nullptr) {
            std::cerr
                << "resident-label access: direct (no query-time flatten copy)\n";
        } else {
            std::cerr
                << "labels staged from .hlbin (excluded; models resident HL): "
                << std::fixed << std::setprecision(6)
                << preload_seconds << " s\n";
        }

        std::cerr
            << "query-time no-sort bucket/work preprocessing: "
            << preprocess_seconds << " s\n"
            << "target hub buckets: "
            << inverted.target_bucket_count << "\n"
            << "common hubs used by sources: "
            << inverted.common_hubs << "\n"
            << "hub-source work items / triple blocks: "
            << inverted.source_work.size() << "\n"
            << "candidate relaxations: "
            << inverted.candidate_relaxations << "\n"
            << "maximum target fanout of a common hub: "
            << inverted.maximum_target_fanout << "\n"
            << "estimated CUDA memory required: "
            << (required_device_bytes / (1024.0 * 1024.0))
            << " MiB; currently free at capacity check: "
            << (free_device_bytes / (1024.0 * 1024.0))
            << " MiB\n"
            << "H2D + CUDA allocation time: "
            << transfer_seconds << " s\n"
            << "result initialization time: "
            << (static_cast<double>(init_ms) / 1000.0) << " s\n"
            << "relaxation kernel time: "
            << (static_cast<double>(relax_ms) / 1000.0) << " s\n"
            << "finalize/validation kernel time: "
            << (static_cast<double>(finalize_ms) / 1000.0) << " s\n"
            << "total CUDA device query time: "
            << gpu_seconds << " s\n"
            << "D2H result/validation time: "
            << download_seconds << " s\n"
            << "TOTAL post-resident-label query time: "
            << post_preload_seconds << " s\n"
            << "average TOTAL post-resident-label time: "
            << std::setprecision(2)
            << average_post_preload_ns << " ns/pair\n"
            << "end-to-end post-resident-label throughput: "
            << end_to_end_pair_throughput << " pairs/s\n"
            << "candidate throughput (relax kernel only): "
            << candidate_throughput << " relaxations/s\n"
            << "result representation: "
            << (distance_only_results
                    ? "32-bit distance-only (quiet hub-centric)"
                    : "64-bit packed distance+hub")
            << "\n"
            << "checksum mode: "
            << (quiet
                    ? "distance-only"
                    : "distance+hub")
            << "\n"
            << "query pairs completed: "
            << pair_count
            << ", unreachable: "
            << unreachable_count
            << ", checksum: "
            << checksum << "\n";
        }

        if (!quiet) {
            std::size_t result_index = 0;

            for (Vertex source : sources) {
                for (Vertex target : targets) {
                    QueryResult result;
                    result.distance =
                        downloaded_results[result_index].distance;
                    result.hub =
                        downloaded_results[result_index].hub;

                    print_query_result(source, target, result);
                    ++result_index;
                }
            }
        }
    } catch (...) {
        if (start_event != nullptr) {
            cudaEventDestroy(start_event);
        }
        if (after_init_event != nullptr) {
            cudaEventDestroy(after_init_event);
        }
        if (after_relax_event != nullptr) {
            cudaEventDestroy(after_relax_event);
        }
        if (end_event != nullptr) {
            cudaEventDestroy(end_event);
        }
        throw;
    }

    return stats;
}

static std::uint64_t derived_benchmark_seed(
    std::uint64_t base_seed,
    std::uint64_t index,
    bool warmup) {

    std::uint64_t value =
        base_seed ^
        (warmup
            ? 0xd1b54a32d192ed03ULL
            : 0x94d049bb133111ebULL);
    value ^= (index + 1ULL) * 0x9e3779b97f4a7c15ULL;
    return mix_query_value(value);
}

static void write_cuda_benchmark_header(std::ostream& out) {
    out
        << "format_version"
        << ",input_file"
        << ",device"
        << ",base_seed"
        << ",run_index"
        << ",query_seed"
        << ",kernel"
        << ",kernel_order"
        << ",cpu_threads"
        << ",source_count"
        << ",target_count"
        << ",pair_count"
        << ",query_generation_s_excluded"
        << ",label_staging_s_excluded"
        << ",estimated_full_hl_bytes"
        << ",persistent_workspace_bytes"
        << ",selected_label_estimated_bytes"
        << ",host_query_structure_peak_bytes"
        << ",host_total_estimated_peak_bytes"
        << ",device_required_bytes"
        << ",device_free_before_bytes"
        << ",target_bucket_count"
        << ",common_hubs"
        << ",source_work_items"
        << ",candidate_relaxations"
        << ",maximum_target_fanout"
        << ",cpu_query_s"
        << ",preprocess_s"
        << ",h2d_allocation_s"
        << ",gpu_init_s"
        << ",gpu_relax_s"
        << ",gpu_finalize_s"
        << ",gpu_total_s"
        << ",d2h_s"
        << ",total_after_staging_s"
        << ",total_ns_per_pair"
        << ",total_pairs_per_s"
        << ",candidate_relaxations_per_s"
        << ",unreachable"
        << ",checksum"
        << "\n";
}

static void write_cuda_benchmark_row(
    std::ostream& out,
    const std::string& input_file,
    const std::string& device_name,
    std::uint64_t base_seed,
    std::uint64_t run_index,
    std::uint64_t query_seed,
    std::uint64_t kernel_order,
    double query_generation_seconds,
    double label_staging_seconds,
    std::uint64_t estimated_full_hl_bytes,
    const CudaQueryBenchmarkStats& stats) {

    /*
     * The full HL is intentionally not included in this process-memory
     * estimate: benchmark mode emulates it as resident by staging only the
     * selected labels before timing.
     */
    std::uint64_t host_total_peak = stats.selected_label_estimated_bytes;
    auto add_host_total = [&](std::uint64_t bytes) {
        if (host_total_peak >
            std::numeric_limits<std::uint64_t>::max() - bytes) {
            host_total_peak = std::numeric_limits<std::uint64_t>::max();
        } else if (host_total_peak !=
                   std::numeric_limits<std::uint64_t>::max()) {
            host_total_peak += bytes;
        }
    };
    add_host_total(stats.persistent_workspace_bytes);
    add_host_total(stats.host_query_structure_bytes);

    const double ns_per_pair =
        stats.pair_count == 0
            ? 0.0
            : stats.total_post_resident_seconds * 1.0e9 /
                static_cast<double>(stats.pair_count);
    const double pairs_per_second =
        stats.total_post_resident_seconds > 0.0
            ? static_cast<double>(stats.pair_count) /
                stats.total_post_resident_seconds
            : 0.0;
    const double relaxations_per_second =
        stats.gpu_relax_seconds > 0.0
            ? static_cast<double>(stats.candidate_relaxations) /
                stats.gpu_relax_seconds
            : 0.0;

    out << std::setprecision(17)
        << 3
        << ',' << csv_escape(input_file)
        << ',' << csv_escape(device_name)
        << ',' << base_seed
        << ',' << run_index
        << ',' << query_seed
        << ',' << stats.kernel
        << ',' << kernel_order
        << ',' << stats.cpu_threads
        << ',' << stats.source_count
        << ',' << stats.target_count
        << ',' << stats.pair_count
        << ',' << query_generation_seconds
        << ',' << label_staging_seconds
        << ',' << estimated_full_hl_bytes
        << ',' << stats.persistent_workspace_bytes
        << ',' << stats.selected_label_estimated_bytes
        << ',' << stats.host_query_structure_bytes
        << ',' << host_total_peak
        << ',' << stats.device_required_bytes
        << ',' << stats.device_free_before_bytes
        << ',' << stats.target_bucket_count
        << ',' << stats.common_hubs
        << ',' << stats.source_work_items
        << ',' << stats.candidate_relaxations
        << ',' << stats.maximum_target_fanout
        << ',' << stats.cpu_query_seconds
        << ',' << stats.preprocess_seconds
        << ',' << stats.h2d_allocation_seconds
        << ',' << stats.gpu_init_seconds
        << ',' << stats.gpu_relax_seconds
        << ',' << stats.gpu_finalize_seconds
        << ',' << stats.gpu_total_seconds
        << ',' << stats.d2h_seconds
        << ',' << stats.total_post_resident_seconds
        << ',' << ns_per_pair
        << ',' << pairs_per_second
        << ',' << relaxations_per_second
        << ',' << stats.unreachable_count
        << ',' << stats.checksum
        << '\n';
}

struct BenchmarkStagedLabels {
    FlatLabelSet source;
    FlatLabelSet target;
    double staging_seconds = 0.0;
};

static void run_random_benchmark(
    DiskHubLabels& labels,
    const std::string& label_path,
    std::size_t source_count,
    std::size_t target_count,
    std::uint64_t run_count,
    std::uint64_t warmup_count,
    std::uint64_t base_seed,
    const std::string& kernel_mode,
    bool include_cpu_pair,
    int cpu_threads,
    int device_id,
    std::uint64_t cache_limit_bytes,
    const std::string& output_path,
    bool append_output) {

    if (source_count == 0 || target_count == 0 || run_count == 0) {
        throw std::runtime_error(
            "Benchmark source count, target count, and run count must be positive");
    }
    if (source_count > labels.num_vertices ||
        target_count > labels.num_vertices) {
        throw std::runtime_error(
            "Benchmark cannot request more unique vertices than the graph contains");
    }
    if (kernel_mode != "pair" &&
        kernel_mode != "hub-source" &&
        kernel_mode != "hub-triple" &&
        kernel_mode != "all" &&
        kernel_mode != "none") {
        throw std::runtime_error(
            "Benchmark CUDA kernel must be pair, hub-source, hub-triple, all, or none");
    }
    if (!include_cpu_pair && kernel_mode == "none") {
        throw std::runtime_error(
            "Benchmark has no selected implementation: use --cpu-pair and/or a CUDA kernel");
    }
    if (cpu_threads <= 0) {
        throw std::runtime_error(
            "Benchmark CPU thread count must be positive");
    }

#ifndef _OPENMP
    if (include_cpu_pair && cpu_threads != 1) {
        throw std::runtime_error(
            "CPU pair benchmark requested multiple threads, but this executable "
            "was built without OpenMP support");
    }
#endif

    const bool use_cuda = kernel_mode != "none";
    cudaDeviceProp properties{};
    std::string gpu_device_name = "not-used";

    if (use_cuda) {
        int device_count = 0;
        cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_id < 0 || device_id >= device_count) {
            throw std::runtime_error(
                "CUDA device ID is out of range; available devices: " +
                std::to_string(device_count));
        }
        cuda_check(cudaSetDevice(device_id), "cudaSetDevice");
        cuda_check(
            cudaGetDeviceProperties(&properties, device_id),
            "cudaGetDeviceProperties");
        gpu_device_name = properties.name;
        cuda_check(cudaFree(nullptr), "CUDA runtime initialization");
    }

    if (include_cpu_pair) {
#ifdef _OPENMP
        omp_set_dynamic(0);
#pragma omp parallel num_threads(cpu_threads)
        {
            // Force OpenMP worker creation before any measured query.
        }
#endif
    }

    const std::uint64_t estimated_full_hl_bytes =
        estimate_resident_label_bytes(labels);

    // This O(|V|) lookup is a reusable query-engine workspace, not
    // query-specific preprocessing. It's allocated once for the whole benchmark.
    std::unique_ptr<HubBucketWorkspace> hub_workspace;
    if (use_cuda &&
        kernel_mode != "pair" &&
        kernel_mode != "none") {
        hub_workspace =
            std::make_unique<HubBucketWorkspace>(labels.num_vertices);
    }

    bool write_header = true;
    if (append_output) {
        std::error_code ec;
        if (std::filesystem::exists(output_path, ec) && !ec) {
            const auto size = std::filesystem::file_size(output_path, ec);
            if (!ec && size != 0) {
                std::ifstream existing(output_path);
                std::string header;
                std::getline(existing, header);
                if (header.find("label_staging_s_excluded") == std::string::npos ||
                    header.find("total_after_staging_s") == std::string::npos ||
                    header.find("cpu_threads") == std::string::npos ||
                    header.find("cpu_query_s") == std::string::npos) {
                    throw std::runtime_error(
                        "--append target uses an older/incompatible benchmark CSV schema; "
                        "use a new output file");
                }
                write_header = false;
            }
        }
    }

    std::ofstream output(
        output_path,
        std::ios::out |
            (append_output ? std::ios::app : std::ios::trunc));
    if (!output) {
        throw std::runtime_error(
            "Cannot open benchmark CSV output: " + output_path);
    }
    output.imbue(std::locale::classic());
    if (write_header) {
        write_cuda_benchmark_header(output);
        output.flush();
    }

    std::vector<std::string> configured_kernels;
    if (include_cpu_pair) {
        configured_kernels.push_back("cpu-pair");
    }
    if (kernel_mode == "all") {
        configured_kernels.push_back("pair");
        configured_kernels.push_back("hub-source");
        configured_kernels.push_back("hub-triple");
    } else if (kernel_mode != "none") {
        configured_kernels.push_back(kernel_mode);
    }

    /*
     * Emulate a fully resident HL on a RAM-limited machine:
     * load exactly the labels needed by this S,T once, before timing any
     * kernel, and reuse the same FlatLabelSet objects for every kernel.
     */
    auto stage_selected_labels = [&](const std::vector<Vertex>& sources,
                                     const std::vector<Vertex>& targets) {
        const std::uint64_t source_bytes =
            estimate_selected_label_bytes(labels.forward, sources);
        const std::uint64_t target_bytes =
            estimate_selected_label_bytes(labels.reverse, targets);

        if (source_bytes == std::numeric_limits<std::uint64_t>::max() ||
            target_bytes == std::numeric_limits<std::uint64_t>::max() ||
            source_bytes >
                std::numeric_limits<std::uint64_t>::max() - target_bytes) {
            throw std::runtime_error(
                "Estimated memory required by staged labels overflows uint64_t");
        }

        const std::uint64_t total_bytes = source_bytes + target_bytes;
        if (total_bytes > cache_limit_bytes) {
            std::ostringstream message;
            message
                << "Selected labels need approximately "
                << std::fixed << std::setprecision(2)
                << (total_bytes / (1024.0 * 1024.0))
                << " MiB, exceeding --cache-mb "
                << (cache_limit_bytes / (1024ULL * 1024ULL))
                << ". Increase --cache-mb.";
            throw std::runtime_error(message.str());
        }

        BenchmarkStagedLabels staged;
        const auto staging_start = std::chrono::steady_clock::now();
        staged.source =
            preload_flat_label_set(
                labels.input,
                labels.forward,
                sources,
                labels.num_vertices,
                "forward label");
        staged.target =
            preload_flat_label_set(
                labels.input,
                labels.reverse,
                targets,
                labels.num_vertices,
                "reverse label");
        const auto staging_end = std::chrono::steady_clock::now();
        staged.staging_seconds =
            std::chrono::duration<double>(
                staging_end - staging_start).count();
        return staged;
    };

    auto execute_kernel = [&](const std::string& kernel,
                              const std::vector<Vertex>& sources,
                              const std::vector<Vertex>& targets,
                              const BenchmarkStagedLabels& staged) {
        if (kernel == "cpu-pair") {
            return run_many_to_many_queries_cpu_pair_staged(
                sources,
                targets,
                staged.source,
                staged.target,
                cpu_threads);
        }
        if (kernel == "pair") {
            return run_many_to_many_queries_cuda(
                labels,
                nullptr,
                sources,
                targets,
                cache_limit_bytes,
                true,
                device_id,
                false,
                &staged.source,
                &staged.target);
        }
        return run_many_to_many_queries_cuda_hub(
            labels,
            nullptr,
            sources,
            targets,
            cache_limit_bytes,
            true,
            device_id,
            kernel,
            false,
            hub_workspace.get(),
            &staged.source,
            &staged.target);
    };

    std::cerr
        << "benchmark: " << run_count << " measured random queries, "
        << warmup_count << " warmup queries, "
        << source_count << " x " << target_count << ", CUDA kernel="
        << kernel_mode
        << ", CPU pair=" << (include_cpu_pair ? "on" : "off") << "\n"
        << "benchmark base seed: " << base_seed << "\n"
        << "benchmark CSV: " << output_path << "\n";
    if (use_cuda) {
        std::cerr
            << "CUDA device " << device_id << ": " << gpu_device_name << "\n";
    }
    if (include_cpu_pair) {
        std::cerr
            << "CPU pair-oriented OpenMP workers requested: "
            << cpu_threads << "\n";
    }
    std::cerr
        << "resident-HL emulation: only selected labels are staged from .hlbin "
           "before timing; full HL is NOT loaded into RAM\n"
        << "estimated full HL size (informational only): "
        << std::fixed << std::setprecision(2)
        << (estimated_full_hl_bytes / (1024.0 * 1024.0 * 1024.0))
        << " GiB\n";

    for (std::uint64_t warmup_index = 0;
         warmup_index < warmup_count;
         ++warmup_index) {
        const std::uint64_t query_seed =
            derived_benchmark_seed(base_seed, warmup_index, true);
        std::mt19937_64 generator(query_seed);
        const std::vector<Vertex> sources = generate_random_vertices(
            source_count, labels.num_vertices, generator);
        const std::vector<Vertex> targets = generate_random_vertices(
            target_count, labels.num_vertices, generator);

        BenchmarkStagedLabels staged =
            stage_selected_labels(sources, targets);

        for (const std::string& kernel : configured_kernels) {
            (void)execute_kernel(kernel, sources, targets, staged);
        }
    }

    for (std::uint64_t run_index = 0;
         run_index < run_count;
         ++run_index) {

        const std::uint64_t query_seed =
            derived_benchmark_seed(base_seed, run_index, false);

        const auto generation_start = std::chrono::steady_clock::now();
        std::mt19937_64 generator(query_seed);
        const std::vector<Vertex> sources = generate_random_vertices(
            source_count, labels.num_vertices, generator);
        const std::vector<Vertex> targets = generate_random_vertices(
            target_count, labels.num_vertices, generator);
        const auto generation_end = std::chrono::steady_clock::now();
        const double query_generation_seconds =
            std::chrono::duration<double>(
                generation_end - generation_start).count();

        BenchmarkStagedLabels staged =
            stage_selected_labels(sources, targets);

        std::vector<std::string> run_kernels = configured_kernels;
        if (run_kernels.size() > 1) {
            const std::size_t shift =
                static_cast<std::size_t>(run_index % run_kernels.size());
            std::rotate(
                run_kernels.begin(),
                run_kernels.begin() + shift,
                run_kernels.end());
        }

        std::vector<CudaQueryBenchmarkStats> measured;
        measured.reserve(run_kernels.size());

        for (const std::string& kernel : run_kernels) {
            measured.push_back(
                execute_kernel(kernel, sources, targets, staged));
        }

        if (measured.size() > 1) {
            const std::uint64_t reference_checksum = measured.front().checksum;
            const std::uint64_t reference_unreachable =
                measured.front().unreachable_count;
            for (const CudaQueryBenchmarkStats& stats : measured) {
                if (stats.checksum != reference_checksum ||
                    stats.unreachable_count != reference_unreachable) {
                    throw std::runtime_error(
                        "Benchmark correctness mismatch between implementations "
                        "for query seed " + std::to_string(query_seed));
                }
            }
        }

        for (std::size_t order = 0; order < measured.size(); ++order) {
            write_cuda_benchmark_row(
                output,
                label_path,
                measured[order].kernel == "cpu-pair"
                    ? std::string("CPU/OpenMP")
                    : gpu_device_name,
                base_seed,
                run_index,
                query_seed,
                static_cast<std::uint64_t>(order),
                query_generation_seconds,
                staged.staging_seconds,
                estimated_full_hl_bytes,
                measured[order]);
        }
        output.flush();

        std::cerr
            << "benchmark run " << (run_index + 1) << '/' << run_count
            << ", query seed " << query_seed
            << ", label staging(excluded)="
            << std::fixed << std::setprecision(3)
            << (staged.staging_seconds * 1000.0) << " ms";
        for (const CudaQueryBenchmarkStats& stats : measured) {
            std::cerr
                << ", " << stats.kernel << '='
                << std::fixed << std::setprecision(3)
                << (stats.total_post_resident_seconds * 1000.0)
                << " ms";
        }
        std::cerr << "\n";
    }

    std::cerr << "benchmark complete; results written to "
              << output_path << "\n";
}

#endif // __CUDACC__

static void print_usage(const char* program) {
    std::cerr
        << "Usage:\n"
        << "  " << program
        << " build [--no-prune|--one-hop|--two-hop]"
           " <input.ch.txt> <output.hlbin>\n"
        << "  " << program
        << " query <input.hlbin> <sources.txt> <targets.txt>"
           " [--backend cpu|cuda] [--cuda-kernel pair|hub-source|hub-triple] [--device N] [--resident-hl] [--cache-mb N] [--threads N] [--quiet]\n"
        << "  " << program
        << " query <input.hlbin> --random N M"
           " [--seed S] [--backend cpu|cuda] [--cuda-kernel pair|hub-source|hub-triple] [--device N] [--resident-hl] [--cache-mb N] [--threads N] [--quiet]\n"
        << "  " << program
        << " benchmark <input.hlbin> --random N M --runs R --output results.csv"
           " [--seed S] [--warmup W] [--cuda-kernel pair|hub-source|hub-triple|all|none]"
           " [--cpu-pair] [--threads N] [--device N] [--cache-mb N] [--append]\n"
        << "\n"
        << "Build pruning modes:\n"
        << "  --no-prune  Disable heuristic pruning\n"
        << "  --one-hop   Use 1-hop pruning (default)\n"
        << "  --two-hop   Use 2-hop pruning\n"
        << "\n"
        << "Query input:\n"
        << "  sources.txt and targets.txt contain whitespace-separated"
           " 1-based vertex IDs.\n"
        << "  --random N M chooses N unique sources and M unique targets"
           " uniformly from the graph.\n"
        << "  --seed S makes a random query reproducible; when omitted,"
           " the generated seed is printed.\n"
        << "  The Cartesian product of sources and targets is queried.\n"
        << "  Selected-label .hlbin staging is excluded to model HL already resident in RAM.\n"
        << "  Query-dependent hub preprocessing, CUDA allocation/H2D, GPU work, and D2H are timed.\n"
        << "  --cache-mb N is the approximate RAM limit for all preloaded"
           " labels (default: 256 MiB).\n"
        << "  --backend cpu|cuda selects the query backend (default: cpu).\n"
        << "  --cuda-kernel pair|hub-source|hub-triple selects the CUDA query mapping\n"
        << "    (default: pair). hub-source uses one thread per (hub,source) occurrence;\n"
        << "    hub-triple builds no-sort target buckets; one block owns (hub,source) and its threads relax targets.\n"
        << "    In --quiet hub-source and hub-triple modes the result matrix is 32-bit distance-only.\n"
        << "  --device N selects the CUDA device (default: 0).\n"
        << "  --resident-hl loads the complete HL arrays into RAM before the query is created;\n"
        << "    query-time selected-label staging is then included in benchmark totals.\n"
        << "  --threads N selects CPU/OpenMP query workers (default:"
           " OpenMP maximum; ignored by CUDA).\n"
        << "  --quiet suppresses per-pair output and is recommended for"
           " benchmarks.\n"
        << "\n"
        << "Benchmark command:\n"
        << "  Does NOT load the complete HL into RAM. For each random S,T, only the\n"
        << "    required forward/reverse labels are staged from .hlbin before timing.\n"
        << "    The staged labels are reused by all selected kernels for that run.\n"
        << "  --runs R writes R measured random query sets to CSV; --warmup W runs W\n"
        << "    unrecorded warmups first (default: 1). Query generation and .hlbin label\n"
        << "    staging are excluded to emulate an already-resident HL index.\n"
        << "  --cuda-kernel all benchmarks CUDA pair, hub-source, and hub-triple on the same\n"
        << "    random S,T. --cuda-kernel none disables CUDA benchmarking.\n"
        << "  --cpu-pair additionally benchmarks pair-oriented CPU/OpenMP HL on the same\n"
        << "    staged labels; --threads N sets the OpenMP worker count. Each pair is an\n"
        << "    independent regular HL merge, executed by a reusable worker thread.\n"
        << "  When multiple implementations are selected, execution order is rotated\n"
        << "    between measured runs.\n"
        << "  --output FILE writes analysis-friendly CSV; --append appends rows and keeps\n"
        << "    an existing header. Each row records phase timings, checksum, work counts,\n"
        << "    selected-label/host memory estimates, and required CUDA memory.\n"
        << "  The CSV total_after_staging_s includes all measured work after selected\n"
        << "    labels are available. For CPU-pair this is the OpenMP query computation;\n"
        << "    for CUDA it includes hub preprocessing (if applicable), allocation/H2D,\n"
        << "    GPU work, and D2H validation/result transfer.\n"
        << "  Output columns in non-quiet query mode are: source target distance hub.\n";
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

        if (command == "build") {
            PruningMode pruning_mode =
                PruningMode::OneHop;

            bool pruning_option_seen = false;
            std::vector<std::string> positional;

            for (int i = 2; i < argc; ++i) {
                const std::string arg = argv[i];

                PruningMode requested_mode{};
                bool is_pruning_option = true;

                if (arg == "--no-prune") {
                    requested_mode = PruningMode::None;
                } else if (arg == "--one-hop") {
                    requested_mode = PruningMode::OneHop;
                } else if (arg == "--two-hop") {
                    requested_mode = PruningMode::TwoHop;
                } else {
                    is_pruning_option = false;
                }

                if (is_pruning_option) {
                    if (pruning_option_seen) {
                        throw std::runtime_error(
                            "Specify only one pruning option");
                    }

                    pruning_mode = requested_mode;
                    pruning_option_seen = true;
                    continue;
                }

                if (arg.rfind("--", 0) == 0) {
                    throw std::runtime_error(
                        "Unknown build option: " + arg);
                }

                positional.push_back(arg);
            }

            if (positional.size() != 2) {
                print_usage(argv[0]);
                return 1;
            }

            const std::string& input_path = positional[0];
            const std::string& output_path = positional[1];

            std::cerr
                << "HL pruning mode: "
                << pruning_mode_name(pruning_mode)
                << "\n";

            CHQueryGraph ch =
                read_ch_text(input_path);

            std::vector<std::uint32_t>().swap(ch.rank);
            std::vector<std::uint32_t>().swap(ch.level);

            build_labels_streaming(
                ch,
                output_path,
                pruning_mode);

            return 0;
        }

        if (command == "benchmark") {
#ifndef __CUDACC__
            throw std::runtime_error(
                "Benchmark command requires an executable built with nvcc");
#else
            if (argc < 4) {
                print_usage(argv[0]);
                return 1;
            }

            constexpr std::uint64_t default_cache_mb = 256;
            std::uint64_t cache_mb = default_cache_mb;
            std::uint64_t run_count = 0;
            std::uint64_t warmup_count = 1;
            std::uint64_t source_count64 = 0;
            std::uint64_t target_count64 = 0;
            std::uint64_t seed = 0;
            bool seed_seen = false;
            bool random_seen = false;
            bool runs_seen = false;
            bool output_seen = false;
            bool warmup_seen = false;
            bool kernel_seen = false;
            bool device_seen = false;
            bool threads_seen = false;
            bool include_cpu_pair = false;
            bool append_output = false;
            std::string output_path;
            std::string cuda_kernel = "all";
            int cuda_device = 0;
#ifdef _OPENMP
            int cpu_threads = omp_get_max_threads();
#else
            int cpu_threads = 1;
#endif

            const std::string label_path = argv[2];

            for (int i = 3; i < argc; ++i) {
                const std::string option = argv[i];

                if (option == "--random") {
                    if (random_seen || i + 2 >= argc) {
                        throw std::runtime_error(
                            "--random must be specified once with N and M");
                    }
                    const std::string n_value = argv[++i];
                    const std::string m_value = argv[++i];
                    std::size_t n_parsed = 0;
                    std::size_t m_parsed = 0;
                    source_count64 = std::stoull(n_value, &n_parsed);
                    target_count64 = std::stoull(m_value, &m_parsed);
                    if (n_parsed != n_value.size() ||
                        m_parsed != m_value.size() ||
                        source_count64 == 0 || target_count64 == 0) {
                        throw std::runtime_error(
                            "--random N M requires two positive integers");
                    }
                    random_seen = true;
                    continue;
                }

                if (option == "--runs") {
                    if (runs_seen || i + 1 >= argc) {
                        throw std::runtime_error(
                            "--runs must be specified once with a positive integer");
                    }
                    const std::string value = argv[++i];
                    std::size_t parsed = 0;
                    run_count = std::stoull(value, &parsed);
                    if (parsed != value.size() || run_count == 0) {
                        throw std::runtime_error(
                            "--runs requires a positive integer");
                    }
                    runs_seen = true;
                    continue;
                }

                if (option == "--warmup") {
                    if (warmup_seen || i + 1 >= argc) {
                        throw std::runtime_error(
                            "--warmup must be specified at most once");
                    }
                    const std::string value = argv[++i];
                    std::size_t parsed = 0;
                    warmup_count = std::stoull(value, &parsed);
                    if (parsed != value.size()) {
                        throw std::runtime_error(
                            "--warmup requires a non-negative integer");
                    }
                    warmup_seen = true;
                    continue;
                }

                if (option == "--output") {
                    if (output_seen || i + 1 >= argc) {
                        throw std::runtime_error(
                            "--output must be specified once with a file path");
                    }
                    output_path = argv[++i];
                    if (output_path.empty()) {
                        throw std::runtime_error(
                            "--output path must not be empty");
                    }
                    output_seen = true;
                    continue;
                }

                if (option == "--seed") {
                    if (seed_seen || i + 1 >= argc) {
                        throw std::runtime_error(
                            "--seed must be specified at most once");
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

                if (option == "--cuda-kernel") {
                    if (kernel_seen || i + 1 >= argc) {
                        throw std::runtime_error(
                            "--cuda-kernel must be specified at most once");
                    }
                    cuda_kernel = argv[++i];
                    if (cuda_kernel != "pair" &&
                        cuda_kernel != "hub-source" &&
                        cuda_kernel != "hub-triple" &&
                        cuda_kernel != "all" &&
                        cuda_kernel != "none") {
                        throw std::runtime_error(
                            "--cuda-kernel must be pair, hub-source, hub-triple, all, or none");
                    }
                    kernel_seen = true;
                    continue;
                }


                if (option == "--cpu-pair") {
                    include_cpu_pair = true;
                    continue;
                }

                if (option == "--threads") {
                    if (threads_seen || i + 1 >= argc) {
                        throw std::runtime_error(
                            "--threads must be specified at most once with a positive integer");
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
                    cpu_threads = static_cast<int>(parsed_threads);
                    threads_seen = true;
                    continue;
                }

                if (option == "--device") {
                    if (device_seen || i + 1 >= argc) {
                        throw std::runtime_error(
                            "--device must be specified at most once");
                    }
                    const std::string value = argv[++i];
                    std::size_t parsed = 0;
                    const std::uint64_t parsed_device =
                        std::stoull(value, &parsed);
                    if (parsed != value.size() ||
                        parsed_device > static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max())) {
                        throw std::runtime_error(
                            "--device must be a non-negative integer that fits in int");
                    }
                    cuda_device = static_cast<int>(parsed_device);
                    device_seen = true;
                    continue;
                }

                if (option == "--cache-mb") {
                    if (i + 1 >= argc) {
                        throw std::runtime_error(
                            "--cache-mb requires a positive integer");
                    }
                    const std::string value = argv[++i];
                    std::size_t parsed = 0;
                    cache_mb = std::stoull(value, &parsed);
                    if (parsed != value.size() || cache_mb == 0) {
                        throw std::runtime_error(
                            "--cache-mb requires a positive integer");
                    }
                    continue;
                }

                if (option == "--append") {
                    append_output = true;
                    continue;
                }

                throw std::runtime_error(
                    "Unknown benchmark option: " + option);
            }

            if (!random_seen || !runs_seen || !output_seen) {
                throw std::runtime_error(
                    "benchmark requires --random N M --runs R --output FILE");
            }

            if (source_count64 >
                    static_cast<std::uint64_t>(
                        std::numeric_limits<std::size_t>::max()) ||
                target_count64 >
                    static_cast<std::uint64_t>(
                        std::numeric_limits<std::size_t>::max())) {
                throw std::runtime_error(
                    "Benchmark source/target counts do not fit in size_t");
            }

            if (cache_mb >
                std::numeric_limits<std::uint64_t>::max() /
                    (1024ULL * 1024ULL)) {
                throw std::runtime_error(
                    "--cache-mb value is too large");
            }
            const std::uint64_t cache_limit_bytes =
                cache_mb * 1024ULL * 1024ULL;

            if (!seed_seen) {
                std::random_device device;
                seed =
                    (static_cast<std::uint64_t>(device()) << 32) ^
                    static_cast<std::uint64_t>(device());
            }

            DiskHubLabels labels = open_labels_for_query(label_path);

            if (source_count64 > labels.num_vertices ||
                target_count64 > labels.num_vertices) {
                throw std::runtime_error(
                    "--random N M cannot request more unique vertices than the graph contains");
            }

            const std::uint64_t resident_bytes =
                estimate_resident_label_bytes(labels);
            std::cerr
                << "benchmark will NOT load the complete HL into RAM; "
                   "estimated full size: "
                << std::fixed << std::setprecision(2)
                << (resident_bytes / (1024.0 * 1024.0 * 1024.0))
                << " GiB\n"
                << "Only the labels required by each random S,T are staged "
                   "from .hlbin before timing and reused by all selected kernels.\n";

            run_random_benchmark(
                labels,
                label_path,
                static_cast<std::size_t>(source_count64),
                static_cast<std::size_t>(target_count64),
                run_count,
                warmup_count,
                seed,
                cuda_kernel,
                include_cpu_pair,
                cpu_threads,
                cuda_device,
                cache_limit_bytes,
                output_path,
                append_output);

            return 0;
#endif
        }

        if (command == "query") {
            if (argc < 4) {
                print_usage(argv[0]);
                return 1;
            }

            constexpr std::uint64_t default_cache_mb = 256;
            std::uint64_t cache_mb = default_cache_mb;
#ifdef _OPENMP
            int requested_threads = omp_get_max_threads();
#else
            int requested_threads = 1;
#endif
            bool threads_seen = false;
            bool quiet = false;
            bool random_mode = false;
            std::string query_backend = "cpu";
            bool backend_seen = false;
            std::string cuda_kernel = "pair";
            bool cuda_kernel_seen = false;
            int cuda_device = 0;
            bool device_seen = false;
            bool resident_hl = false;
            bool seed_seen = false;
            std::uint64_t seed = 0;
            std::uint64_t random_source_count64 = 0;
            std::uint64_t random_target_count64 = 0;
            std::vector<std::string> positional;

            const std::string label_path = argv[2];

            for (int i = 3; i < argc; ++i) {
                const std::string option = argv[i];

                if (option == "--backend") {
                    if (i + 1 >= argc) {
                        throw std::runtime_error(
                            "--backend requires cpu or cuda");
                    }
                    if (backend_seen) {
                        throw std::runtime_error(
                            "Specify --backend at most once");
                    }

                    query_backend = argv[++i];
                    if (query_backend != "cpu" &&
                        query_backend != "cuda") {
                        throw std::runtime_error(
                            "--backend must be cpu or cuda");
                    }
                    backend_seen = true;
                    continue;
                }

                if (option == "--cuda-kernel") {
                    if (i + 1 >= argc) {
                        throw std::runtime_error(
                            "--cuda-kernel requires pair, hub-source, or hub-triple");
                    }
                    if (cuda_kernel_seen) {
                        throw std::runtime_error(
                            "Specify --cuda-kernel at most once");
                    }

                    cuda_kernel = argv[++i];
                    if (cuda_kernel != "pair" &&
                        cuda_kernel != "hub-source" &&
                        cuda_kernel != "hub-triple") {
                        throw std::runtime_error(
                            "--cuda-kernel must be pair, hub-source, or hub-triple");
                    }
                    cuda_kernel_seen = true;
                    continue;
                }

                if (option == "--device") {
                    if (i + 1 >= argc) {
                        throw std::runtime_error(
                            "--device requires a non-negative integer");
                    }
                    if (device_seen) {
                        throw std::runtime_error(
                            "Specify --device at most once");
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
                            "--device must be a non-negative integer that fits in int");
                    }

                    cuda_device = static_cast<int>(parsed_device);
                    device_seen = true;
                    continue;
                }

                if (option == "--resident-hl") {
                    resident_hl = true;
                    continue;
                }

                if (option == "--quiet") {
                    quiet = true;
                    continue;
                }

                if (option == "--cache-mb") {
                    if (i + 1 >= argc) {
                        throw std::runtime_error(
                            "--cache-mb requires a numeric value");
                    }

                    const std::string value = argv[++i];
                    std::size_t parsed = 0;
                    cache_mb = std::stoull(value, &parsed);

                    if (parsed != value.size() || cache_mb == 0) {
                        throw std::runtime_error(
                            "--cache-mb must be a positive integer");
                    }
                    continue;
                }

                if (option == "--threads") {
                    if (i + 1 >= argc) {
                        throw std::runtime_error(
                            "--threads requires a positive integer value");
                    }
                    if (threads_seen) {
                        throw std::runtime_error(
                            "Specify --threads at most once");
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

                    requested_threads =
                        static_cast<int>(parsed_threads);
                    threads_seen = true;
                    continue;
                }

                if (option == "--seed") {
                    if (i + 1 >= argc) {
                        throw std::runtime_error(
                            "--seed requires an integer value");
                    }
                    if (seed_seen) {
                        throw std::runtime_error(
                            "Specify --seed at most once");
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
                    if (random_mode) {
                        throw std::runtime_error(
                            "Specify --random at most once");
                    }
                    if (i + 2 >= argc) {
                        throw std::runtime_error(
                            "--random requires N and M");
                    }

                    const std::string n_value = argv[++i];
                    const std::string m_value = argv[++i];
                    std::size_t n_parsed = 0;
                    std::size_t m_parsed = 0;

                    random_source_count64 =
                        std::stoull(n_value, &n_parsed);
                    random_target_count64 =
                        std::stoull(m_value, &m_parsed);

                    if (n_parsed != n_value.size() ||
                        m_parsed != m_value.size() ||
                        random_source_count64 == 0 ||
                        random_target_count64 == 0) {
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

            if (device_seen && query_backend != "cuda") {
                throw std::runtime_error(
                    "--device is only valid with --backend cuda");
            }

            if (cuda_kernel_seen && query_backend != "cuda") {
                throw std::runtime_error(
                    "--cuda-kernel is only valid with --backend cuda");
            }

            if (resident_hl && query_backend != "cuda") {
                throw std::runtime_error(
                    "--resident-hl is currently only valid with --backend cuda");
            }

#ifndef __CUDACC__
            if (query_backend == "cuda") {
                throw std::runtime_error(
                    "CUDA backend requested, but this executable was not built with nvcc");
            }
#endif

            if (cache_mb >
                std::numeric_limits<std::uint64_t>::max() /
                    (1024ULL * 1024ULL)) {
                throw std::runtime_error(
                    "--cache-mb value is too large");
            }

            const std::uint64_t cache_limit_bytes =
                cache_mb * 1024ULL * 1024ULL;

            DiskHubLabels labels =
                open_labels_for_query(label_path);

            std::unique_ptr<ResidentHubLabels> resident_labels;

            if (resident_hl) {
                const std::uint64_t resident_bytes =
                    estimate_resident_label_bytes(labels);

                std::cerr
                    << "loading complete HL into RAM before query; estimated resident size: "
                    << std::fixed << std::setprecision(2)
                    << (resident_bytes / (1024.0 * 1024.0 * 1024.0))
                    << " GiB\n";

                const auto resident_load_start =
                    std::chrono::steady_clock::now();

                resident_labels =
                    std::make_unique<ResidentHubLabels>(
                        load_labels_resident(labels));

                const auto resident_load_end =
                    std::chrono::steady_clock::now();

                std::cerr
                    << "complete HL resident in RAM; one-time load (not benchmarked): "
                    << std::chrono::duration<double>(
                        resident_load_end - resident_load_start).count()
                    << " s\n";
            }

            std::vector<Vertex> sources;
            std::vector<Vertex> targets;

            if (random_mode) {
                if (!positional.empty()) {
                    throw std::runtime_error(
                        "Do not specify source/target files together with --random");
                }

                if (random_source_count64 > labels.num_vertices ||
                    random_target_count64 > labels.num_vertices) {
                    throw std::runtime_error(
                        "--random N M cannot request more unique vertices "
                        "than exist in the graph");
                }

                if (random_source_count64 >
                        static_cast<std::uint64_t>(
                            std::numeric_limits<std::size_t>::max()) ||
                    random_target_count64 >
                        static_cast<std::uint64_t>(
                            std::numeric_limits<std::size_t>::max())) {
                    throw std::runtime_error(
                        "Random query counts do not fit in size_t");
                }

                if (!seed_seen) {
                    std::random_device device;
                    seed =
                        (static_cast<std::uint64_t>(device()) << 32) ^
                        static_cast<std::uint64_t>(device());
                }

                std::cerr
                    << "random query seed: "
                    << seed
                    << "\n";

                std::mt19937_64 generator(seed);

                sources = generate_random_vertices(
                    static_cast<std::size_t>(random_source_count64),
                    labels.num_vertices,
                    generator);

                targets = generate_random_vertices(
                    static_cast<std::size_t>(random_target_count64),
                    labels.num_vertices,
                    generator);
            } else {
                if (seed_seen) {
                    throw std::runtime_error(
                        "--seed is only valid together with --random");
                }

                if (positional.size() != 2) {
                    print_usage(argv[0]);
                    return 1;
                }

                sources = read_vertex_id_file(
                    positional[0],
                    labels.num_vertices,
                    "source");

                targets = read_vertex_id_file(
                    positional[1],
                    labels.num_vertices,
                    "target");
            }

            if (query_backend == "cpu") {
                run_many_to_many_queries_cpu(
                    labels,
                    sources,
                    targets,
                    cache_limit_bytes,
                    quiet,
                    requested_threads);
            } else {
#ifdef __CUDACC__
                if (cuda_kernel == "pair") {
                    run_many_to_many_queries_cuda(
                        labels,
                        resident_labels.get(),
                        sources,
                        targets,
                        cache_limit_bytes,
                        quiet,
                        cuda_device);
                } else {
                    run_many_to_many_queries_cuda_hub(
                        labels,
                        resident_labels.get(),
                        sources,
                        targets,
                        cache_limit_bytes,
                        quiet,
                        cuda_device,
                        cuda_kernel);
                }
#else
                throw std::runtime_error(
                    "CUDA backend requested, but CUDA support is unavailable");
#endif
            }

            return 0;
        }

        print_usage(argv[0]);
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}

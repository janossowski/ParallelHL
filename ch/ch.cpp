#include <bits/stdc++.h>
#include <atomic>

using namespace std;
#ifdef _OPENMP
#include <omp.h>
#endif

using Weight = uint64_t;
static constexpr Weight INF = numeric_limits<Weight>::max() / 4;

struct Arc {
    int from;
    int to;
    Weight w;
    uint32_t hops;   // number of original arcs represented
    int mid;         // middle vertex for unpacking; -1 for original arcs
    bool shortcut;
};

static inline uint64_t arc_key(int u, int v) {
    return (uint64_t(uint32_t(u)) << 32) | uint32_t(v);
}

struct DynGraph {
    int n = 0;
    vector<Arc> arcs;
    vector<vector<int>> out, in;
    unordered_map<uint64_t, int> arc_index;

    DynGraph() = default;
    explicit DynGraph(int n_) : n(n_), out(n_), in(n_) {}

    int add_or_relax(int u, int v, Weight w,
                     uint32_t hops = 1,
                     int mid = -1,
                     bool shortcut = false) {
        if (u == v) return -1;

        uint64_t k = arc_key(u, v);
        auto it = arc_index.find(k);

        if (it != arc_index.end()) {
            int id = it->second;
            if (w < arcs[id].w) {
                arcs[id].w = w;
                arcs[id].hops = hops;
                arcs[id].mid = mid;
                arcs[id].shortcut = shortcut;
            }
            return id;
        }

        int id = static_cast<int>(arcs.size());
        arcs.push_back({u, v, w, hops, mid, shortcut});
        out[u].push_back(id);
        in[v].push_back(id);
        arc_index.emplace(k, id);
        return id;
    }
};

struct CHTextWriteOptions {
    bool include_augmented_arcs = true;

    // Progress message every N vertices/arcs/edges written.
    uint64_t progress_every = 5'000'000;
    bool binary_mode_newlines = true;
};

DynGraph read_dimacs_gr(istream& input) {
    string line;
    DynGraph g;

    while (getline(input, line)) {
        if (line.empty()) continue;

        size_t first = line.find_first_not_of(" \t\r\n");
        if (first == string::npos) continue;
        if (line.compare(first, 2, "//") == 0) continue;

        char kind = line[first];
        if (kind == 'c') continue;

        stringstream ss(line.substr(first));

        if (kind == 'p') {
            string p, sp;
            int n;
            size_t m;
            ss >> p >> sp >> n >> m;

            g = DynGraph(n);
            g.arcs.reserve(m);
            g.arc_index.reserve(m * 2 + 1);
        } else if (kind == 'a') {
            char a;
            int u, v;
            Weight w;
            ss >> a >> u >> v >> w;

            if (u <= 0 || v <= 0 || u > g.n || v > g.n) {
                throw runtime_error("arc endpoint outside p sp node range");
            }

            g.add_or_relax(u - 1, v - 1, w, 1, -1, false);
        }
    }

    return g;
}

using AdjList = vector<vector<pair<int, Weight>>>;

AdjList build_original_adjacency(const DynGraph& g) {
    AdjList adj(g.n);

    for (const Arc& e : g.arcs) {
        if (!e.shortcut) {
            adj[e.from].push_back({e.to, e.w});
        }
    }

    return adj;
}

string format_distance(Weight d) {
    if (d >= INF / 2) return "INF";
    return to_string(d);
}

bool same_distance(Weight a, Weight b) {
    if (a >= INF / 2 && b >= INF / 2) return true;
    return a == b;
}

struct DijkstraWorkspace {
    vector<Weight> dist;
    vector<int> touched;

    explicit DijkstraWorkspace(int n = 0) : dist(n, INF) {}

    void ensure_size(int n) {
        if (static_cast<int>(dist.size()) == n) return;
        dist.assign(n, INF);
        touched.clear();
    }

    Weight run(const AdjList& adj, int source, int target, size_t* settled_out = nullptr) {
        int n = static_cast<int>(adj.size());

        if (source < 0 || source >= n || target < 0 || target >= n) {
            throw out_of_range("Dijkstra query vertex out of range");
        }

        ensure_size(n);

        using Item = pair<Weight, int>;
        priority_queue<Item, vector<Item>, greater<Item>> pq;

        auto set_dist = [&](int v, Weight d) {
            if (dist[v] == INF) touched.push_back(v);
            dist[v] = d;
            pq.push({d, v});
        };

        size_t settled = 0;
        Weight answer = INF;

        set_dist(source, 0);

        while (!pq.empty()) {
            auto [d, u] = pq.top();
            pq.pop();

            if (d != dist[u]) continue;

            ++settled;

            if (u == target) {
                answer = d;
                break;
            }

            for (auto [v, w] : adj[u]) {
                if (d > INF - w) continue;

                Weight nd = d + w;

                if (nd < dist[v]) {
                    set_dist(v, nd);
                }
            }
        }

        for (int v : touched) {
            dist[v] = INF;
        }

        touched.clear();

        if (settled_out != nullptr) {
            *settled_out = settled;
        }

        return answer;
    }
};

struct CHQueryResult {
    Weight distance = INF;
    size_t settled_forward = 0;
    size_t settled_reverse = 0;
};

struct CHQueryWorkspace {
    vector<Weight> dist_f;
    vector<Weight> dist_b;
    vector<int> seen_f;
    vector<int> seen_b;

    explicit CHQueryWorkspace(int n = 0)
        : dist_f(n, INF), dist_b(n, INF) {}

    void ensure_size(int n) {
        if (static_cast<int>(dist_f.size()) == n) return;

        dist_f.assign(n, INF);
        dist_b.assign(n, INF);
        seen_f.clear();
        seen_b.clear();
    }

    void reset() {
        for (int v : seen_f) dist_f[v] = INF;
        for (int v : seen_b) dist_b[v] = INF;

        seen_f.clear();
        seen_b.clear();
    }
};

struct WitnessSearch {
    vector<Weight> dist;
    vector<int> touched;

    explicit WitnessSearch(int n = 0) : dist(n, INF) {}

    Weight run(const DynGraph& g,
               const vector<char>& contracted,
               int source,
               int target,
               int forbidden,
               Weight cutoff) {
        using Item = pair<Weight, int>;
        priority_queue<Item, vector<Item>, greater<Item>> pq;

        auto set_dist = [&](int v, Weight d) {
            if (dist[v] == INF) touched.push_back(v);
            dist[v] = d;
            pq.push({d, v});
        };

        set_dist(source, 0);
        Weight answer = INF;

        while (!pq.empty()) {
            auto [d, u] = pq.top();
            pq.pop();

            if (d != dist[u]) continue;
            if (d > cutoff) break;

            if (u == target) {
                answer = d;
                break;
            }

            for (int eid : g.out[u]) {
                const Arc& e = g.arcs[eid];
                int v = e.to;

                if (v == forbidden || contracted[v]) continue;
                if (d > cutoff - e.w) continue;

                Weight nd = d + e.w;
                if (nd < dist[v]) {
                    set_dist(v, nd);
                }
            }
        }

        for (int v : touched) dist[v] = INF;
        touched.clear();

        return answer;
    }
};

struct ShortcutCandidate {
    int from;
    int to;
    Weight w;
    uint32_t hops;
    int mid;
};

struct PriorityEval {
    long long priority = 0;
    int edge_difference = 0;
    int contracted_neighbors = 0;
    int assigned_level = 0;
    uint64_t added_hops = 0;
    vector<ShortcutCandidate> shortcuts;
};

struct ContractionLogOptions {
    bool enabled = true;

    // If true, prints a line for every contracted vertex
    bool log_each_vertex = false;

    // Prints a progress summary every N contractions
    // Ignored if log_each_vertex is true
    size_t log_every = 10000;

    ostream* out = &cerr;
};

using EdgeWeight = uint32_t;

struct QueryEdge {
    int to;
    EdgeWeight w;
};

struct CSRGraph {
    int n = 0;
    vector<uint64_t> offset;
    unique_ptr<QueryEdge[]> edges;
    uint64_t m = 0;

    void clear() {
        n = 0;
        offset.clear();
        edges.reset();
        m = 0;
    }

    void allocate_edges(uint64_t count) {
        m = count;

        if (count == 0) {
            edges.reset();
            return;
        }

        edges.reset(new QueryEdge[count]);
    }

    uint64_t edge_count() const {
        return m;
    }
};

class ContractionHierarchy {
public:
    explicit ContractionHierarchy(DynGraph graph_)
        : graph(std::move(graph_)), n(graph.n) {
        rank.assign(n, -1);
        level.assign(n, 0);
        contracted.assign(n, false);

        set_witness_threads(1);
    }

    void set_witness_threads(int threads) {
    #ifdef _OPENMP
        if (threads <= 0) {
            threads = omp_get_max_threads();
        }
    #else
        (void)threads;
        threads = 1;
    #endif

        witness_threads = max(1, threads);

        witness_pool.clear();
        witness_pool.reserve(witness_threads);

        for (int i = 0; i < witness_threads; ++i) {
            witness_pool.push_back(make_unique<WitnessSearch>(n));
        }
    }

    void set_parallel_witness_min_pairs(size_t min_pairs) {
        parallel_witness_min_pairs = max<size_t>(1, min_pairs);
    }

    int get_witness_threads() const {
        return witness_threads;
    }

    void preprocess(const ContractionLogOptions& log = {}) {
        using Clock = chrono::steady_clock;

        struct QItem {
            long long priority;
            int vertex;

            bool operator>(const QItem& other) const {
                if (priority != other.priority) return priority > other.priority;
                return vertex > other.vertex;
            }
        };

        auto start_time = Clock::now();

        auto elapsed_seconds = [&]() -> double {
            auto now = Clock::now();
            return chrono::duration<double>(now - start_time).count();
        };

        auto print_progress = [&](int done,
                                int vertex,
                                const PriorityEval& ev,
                                size_t shortcuts_added,
                                bool final_line = false) {
            if (!log.enabled || log.out == nullptr) return;

            double elapsed = elapsed_seconds();
            double pct = n == 0 ? 100.0 : (100.0 * double(done) / double(n));
            double rate = elapsed > 0.0 ? double(done) / elapsed : 0.0;
            double remaining = rate > 0.0 ? double(n - done) / rate : 0.0;

            (*log.out)
                << "[CH] "
                << (final_line ? "done " : "progress ")
                << done << "/" << n
                << " (" << fixed << setprecision(2) << pct << "%)"
                << " elapsed=" << setprecision(1) << elapsed << "s";

            if (!final_line) {
                (*log.out)
                    << " eta=" << setprecision(1) << remaining << "s"
                    << " rate=" << setprecision(0) << rate << " vertices/s"
                    << " vertex=" << (vertex + 1)
                    << " rank=" << rank[vertex]
                    << " priority=" << ev.priority
                    << " ED=" << ev.edge_difference
                    << " CN=" << ev.contracted_neighbors
                    << " H=" << ev.added_hops
                    << " L=" << ev.assigned_level
                    << " shortcuts=" << shortcuts_added
                    << " total_arcs=" << graph.arcs.size();
            }

            (*log.out) << "\n";
        };

        priority_queue<QItem, vector<QItem>, greater<QItem>> pq;

        if (log.enabled && log.out != nullptr) {
            (*log.out) << "[CH] initial priority computation for " << n << " vertices...\n";
        }

        for (int v = 0; v < n; ++v) {
            PriorityEval ev = evaluate(v);
            pq.push({ev.priority, v});

            if (log.enabled && log.out != nullptr && log.log_every > 0) {
                if ((v + 1) % log.log_every == 0 || v + 1 == n) {
                    (*log.out)
                        << "[CH] initialized priorities "
                        << (v + 1) << "/" << n
                        << " (" << fixed << setprecision(2)
                        << (100.0 * double(v + 1) / double(n)) << "%)\n";
                }
            }
        }

        if (log.enabled && log.out != nullptr) {
            (*log.out) << "[CH] starting contractions...\n";
        }

        int next_rank = 0;

        while (!pq.empty()) {
            QItem item = pq.top();
            pq.pop();

            int v = item.vertex;
            if (contracted[v]) continue;

            PriorityEval ev = evaluate(v);

            // Reinsert if the priority changed since insertion
            if (ev.priority != item.priority) {
                pq.push({ev.priority, v});
                continue;
            }

            size_t shortcuts_added = 0;

            for (const auto& sc : ev.shortcuts) {
                int id = graph.add_or_relax(sc.from, sc.to, sc.w, sc.hops, sc.mid, true);
                if (id >= 0) {
                    ++shortcuts_added;
                }
            }

            contracted[v] = true;
            rank[v] = next_rank++;
            level[v] = ev.assigned_level;

            vector<int> affected;
            collect_incident_uncontracted_vertices(v, affected);

            for (const auto& sc : ev.shortcuts) {
                if (!contracted[sc.from]) affected.push_back(sc.from);
                if (!contracted[sc.to]) affected.push_back(sc.to);
            }

            sort(affected.begin(), affected.end());
            affected.erase(unique(affected.begin(), affected.end()), affected.end());

            for (int u : affected) {
                PriorityEval neu = evaluate(u);
                pq.push({neu.priority, u});
            }

            int done = next_rank;

            bool should_log =
                log.enabled &&
                log.out != nullptr &&
                (
                    log.log_each_vertex ||
                    done == n ||
                    (log.log_every > 0 && done % static_cast<int>(log.log_every) == 0)
                );

            if (should_log) {
                print_progress(done, v, ev, shortcuts_added, done == n);
            }
        }

        if (log.enabled && log.out != nullptr) {
            (*log.out) << "[CH] building upward/downward query graphs...\n";
        }

        build_query_graphs();

        if (log.enabled && log.out != nullptr) {
            (*log.out)
                << "[CH] preprocessing complete. augmented_arcs="
                << graph.arcs.size()
                << " elapsed=" << fixed << setprecision(1)
                << elapsed_seconds() << "s\n";
        }
    }

    int vertex_count() const {
        return n;
    }

    CHQueryResult query_distance(int s, int t, CHQueryWorkspace& ws) const {
        if (s < 0 || s >= n || t < 0 || t >= n) {
            throw out_of_range("CH query vertex out of range");
        }

        ws.ensure_size(n);
        ws.reset();

        CHQueryResult result;

        result.settled_forward = run_dijkstra_all(s, upward, ws.dist_f, ws.seen_f);

        result.settled_reverse = run_dijkstra_all(t, reverse_downward, ws.dist_b, ws.seen_b);

        Weight best = INF;

        for (int v : ws.seen_f) {
            if (ws.dist_b[v] == INF) continue;

            if (ws.dist_f[v] <= INF - ws.dist_b[v]) {
                best = min(best, ws.dist_f[v] + ws.dist_b[v]);
            }
        }

        result.distance = best;

        ws.reset();

        return result;
    }

    Weight query_distance(int s, int t) const {
        CHQueryWorkspace ws(n);
        return query_distance(s, t, ws).distance;
    }

    void set_query_graph_threads(int threads) {
    #ifdef _OPENMP
        int max_threads = omp_get_max_threads();

        if (threads <= 0) {
            threads = max_threads;
        }

        query_graph_threads = max(1, min(threads, max_threads));
    #else
        (void)threads;
        query_graph_threads = 1;
    #endif
    }

    int get_query_graph_threads() const {
        return query_graph_threads;
    }

    void set_query_graph_progress(bool enabled,
                                uint64_t every_vertices = 1'000'000,
                                uint64_t chunk_vertices = 65'536) {
        query_graph_progress_enabled = enabled;
        query_graph_progress_every = max<uint64_t>(1, every_vertices);
        query_graph_chunk_vertices = max<uint64_t>(1, chunk_vertices);
    }

    void write_ch_text(const string& path,
                    const CHTextWriteOptions& opts = {}) const {
        write_ch_text_impl(path, opts);
    }

private:
    DynGraph graph;
    int n;

    int witness_threads = 1;
    size_t parallel_witness_min_pairs = 32;

    vector<unique_ptr<WitnessSearch>> witness_pool; 

    vector<int> rank;
    vector<int> level;
    vector<char> contracted;

    CSRGraph upward;
    CSRGraph reverse_downward;

    int query_graph_threads = 1;
    bool query_graph_progress_enabled = true;
    uint64_t query_graph_progress_every = 1'000'000;
    uint64_t query_graph_chunk_vertices = 65'536;

    vector<int> active_in_arcs(int v) const {
        vector<int> ids;

        for (int eid : graph.in[v]) {
            const Arc& e = graph.arcs[eid];
            if (!contracted[e.from] && e.from != v) {
                ids.push_back(eid);
            }
        }

        return ids;
    }

    vector<int> active_out_arcs(int v) const {
        vector<int> ids;

        for (int eid : graph.out[v]) {
            const Arc& e = graph.arcs[eid];
            if (!contracted[e.to] && e.to != v) {
                ids.push_back(eid);
            }
        }

        return ids;
    }

    void collect_incident_uncontracted_vertices(int v, vector<int>& result) const {
        for (int eid : graph.in[v]) {
            int u = graph.arcs[eid].from;
            if (u != v && !contracted[u]) result.push_back(u);
        }

        for (int eid : graph.out[v]) {
            int u = graph.arcs[eid].to;
            if (u != v && !contracted[u]) result.push_back(u);
        }
    }

    vector<int> incident_contracted_neighbors(int v) const {
        vector<int> nbrs;

        for (int eid : graph.in[v]) {
            int u = graph.arcs[eid].from;
            if (u != v && contracted[u]) nbrs.push_back(u);
        }

        for (int eid : graph.out[v]) {
            int u = graph.arcs[eid].to;
            if (u != v && contracted[u]) nbrs.push_back(u);
        }

        sort(nbrs.begin(), nbrs.end());
        nbrs.erase(unique(nbrs.begin(), nbrs.end()), nbrs.end());

        return nbrs;
    }

    int would_assign_level(int v) const {
        auto nbrs = incident_contracted_neighbors(v);

        if (nbrs.empty()) return 0;

        int best = 0;
        for (int u : nbrs) {
            best = max(best, level[u] + 1);
        }

        return best;
    }

    PriorityEval evaluate(int v) {
        PriorityEval ev;

        ev.assigned_level = would_assign_level(v);

        auto contracted_nbrs = incident_contracted_neighbors(v);
        ev.contracted_neighbors = static_cast<int>(contracted_nbrs.size());

        auto ins = active_in_arcs(v);
        auto outs = active_out_arcs(v);

        int removed_arcs = static_cast<int>(ins.size() + outs.size());

        struct WitnessTask {
            int in_id;
            int out_id;
        };

        vector<WitnessTask> tasks;
        tasks.reserve(ins.size() * max<size_t>(1, outs.size()));

        for (int in_id : ins) {
            const Arc& a = graph.arcs[in_id];
            int u = a.from;

            for (int out_id : outs) {
                const Arc& b = graph.arcs[out_id];
                int w = b.to;

                if (u == w) continue;
                if (a.w > INF - b.w) continue;

                tasks.push_back({in_id, out_id});
            }
        }

        int actual_threads = 1;

    #ifdef _OPENMP
        if (witness_threads > 1 && tasks.size() >= parallel_witness_min_pairs) {
            actual_threads = witness_threads;
        }
    #endif

        vector<vector<ShortcutCandidate>> local_candidates(actual_threads);

        auto process_range = [&](size_t begin, size_t end, int tid) {
            WitnessSearch& local_witness = *witness_pool[tid];
            auto& output = local_candidates[tid];

            for (size_t i = begin; i < end; ++i) {
                const Arc& a = graph.arcs[tasks[i].in_id];
                const Arc& b = graph.arcs[tasks[i].out_id];

                int u = a.from;
                int w = b.to;

                Weight via = a.w + b.w;

                // Exact witness search in the current graph, avoiding v
                Weight witness_dist =
                    local_witness.run(graph, contracted, u, w, v, via);

                if (witness_dist <= via) continue;

                uint64_t hops64 = uint64_t(a.hops) + uint64_t(b.hops);

                uint32_t hops =
                    hops64 > numeric_limits<uint32_t>::max()
                    ? numeric_limits<uint32_t>::max()
                    : uint32_t(hops64);

                output.push_back(ShortcutCandidate{u, w, via, hops, v});
            }
        };

    #ifdef _OPENMP
        if (actual_threads > 1) {
            #pragma omp parallel num_threads(actual_threads)
            {
                int tid = omp_get_thread_num();

                size_t total = tasks.size();
                size_t begin = (total * size_t(tid)) / size_t(actual_threads);
                size_t end = (total * size_t(tid + 1)) / size_t(actual_threads);

                process_range(begin, end, tid);
            }
        } else
    #endif
        {
            process_range(0, tasks.size(), 0);
        }

        unordered_map<uint64_t, ShortcutCandidate> best_pair;

        size_t candidate_count = 0;
        for (const auto& bucket : local_candidates) {
            candidate_count += bucket.size();
        }

        best_pair.reserve(candidate_count * 2 + 1);

        for (const auto& bucket : local_candidates) {
            for (const ShortcutCandidate& sc : bucket) {
                uint64_t k = arc_key(sc.from, sc.to);

                auto it = best_pair.find(k);

                if (it == best_pair.end() || sc.w < it->second.w) {
                    best_pair[k] = sc;
                }
            }
        }

        ev.shortcuts.reserve(best_pair.size());

        for (auto& [_, sc] : best_pair) {
            ev.added_hops += sc.hops;
            ev.shortcuts.push_back(sc);
        }

        ev.edge_difference =
            static_cast<int>(ev.shortcuts.size()) - removed_arcs;

        __int128 p =
            2LL * ev.edge_difference +
            ev.contracted_neighbors +
            (__int128)ev.added_hops +
            5LL * ev.assigned_level;

        if (p > numeric_limits<long long>::max()) {
            ev.priority = numeric_limits<long long>::max();
        } else if (p < numeric_limits<long long>::min()) {
            ev.priority = numeric_limits<long long>::min();
        } else {
            ev.priority = static_cast<long long>(p);
        }

        return ev;
    }

    void build_query_graphs() {
        using Clock = chrono::steady_clock;

        auto total_start = Clock::now();

        int threads = max(1, query_graph_threads);

    #ifdef _OPENMP
        threads = min(threads, omp_get_max_threads());
    #else
        threads = 1;
    #endif

        if (query_graph_progress_enabled) {
            cerr << "[CH-CSR] building upward/reverse-downward query graphs"
                << " n=" << n
                << " augmented_arcs=" << graph.arcs.size()
                << " threads=" << threads
                << "\n";
        }

        upward.clear();
        reverse_downward.clear();

        upward.n = n;
        reverse_downward.n = n;

        vector<uint64_t> up_count(n, 0);
        vector<uint64_t> rev_down_count(n, 0);

        run_query_graph_vertex_phase(
            "count upward degrees",
            threads,
            [&](int begin, int end) {
                for (int u = begin; u < end; ++u) {
                    uint64_t count = 0;
                    int ru = rank[u];

                    for (int eid : graph.out[u]) {
                        const Arc& e = graph.arcs[eid];
                        int rv = rank[e.to];

                        if (ru < 0 || rv < 0 || ru == rv) continue;

                        if (ru < rv) {
                            ++count;
                        }
                    }

                    up_count[u] = count;
                }
            }
        );

        run_query_graph_vertex_phase(
            "count reverse-downward degrees",
            threads,
            [&](int begin, int end) {
                for (int v = begin; v < end; ++v) {
                    uint64_t count = 0;
                    int rv = rank[v];

                    for (int eid : graph.in[v]) {
                        const Arc& e = graph.arcs[eid];
                        int ru = rank[e.from];

                        if (ru < 0 || rv < 0 || ru == rv) continue;

                        if (ru > rv) {
                            ++count;
                        }
                    }

                    rev_down_count[v] = count;
                }
            }
        );

        if (query_graph_progress_enabled) {
            cerr << "[CH-CSR] prefix sums started\n";
        }

        auto prefix_start = Clock::now();

        upward.offset.assign(static_cast<size_t>(n) + 1, 0);
        reverse_downward.offset.assign(static_cast<size_t>(n) + 1, 0);

        for (int v = 0; v < n; ++v) {
            upward.offset[v + 1] = upward.offset[v] + up_count[v];
            reverse_downward.offset[v + 1] =
                reverse_downward.offset[v] + rev_down_count[v];
        }

        uint64_t upward_edges = upward.offset[n];
        uint64_t reverse_downward_edges = reverse_downward.offset[n];

        auto prefix_elapsed =
            chrono::duration<double>(Clock::now() - prefix_start).count();

        if (query_graph_progress_enabled) {
            cerr << "[CH-CSR] prefix sums complete"
                << " upward_edges=" << upward_edges
                << " reverse_downward_edges=" << reverse_downward_edges
                << " elapsed=" << fixed << setprecision(1)
                << prefix_elapsed << "s"
                << "\n";

            long double bytes =
                static_cast<long double>(upward.offset.size()) * sizeof(uint64_t) +
                static_cast<long double>(reverse_downward.offset.size()) * sizeof(uint64_t) +
                static_cast<long double>(upward_edges) * sizeof(QueryEdge) +
                static_cast<long double>(reverse_downward_edges) * sizeof(QueryEdge);

            cerr << "[CH-CSR] allocating query graph storage"
                << " approx="
                << fixed << setprecision(2)
                << static_cast<double>(bytes / (1024.0L * 1024.0L * 1024.0L))
                << " GiB"
                << "\n";
        }

    auto alloc_start = Clock::now();

    try {
        auto up_alloc_start = Clock::now();
        cerr << "[CH-CSR] allocating upward.edges: "
            << upward_edges << " edges, approx "
            << fixed << setprecision(2)
            << (static_cast<long double>(upward_edges) * sizeof(QueryEdge)
                / (1024.0L * 1024.0L * 1024.0L))
            << " GiB\n";

        upward.allocate_edges(upward_edges);

        cerr << "[CH-CSR] upward.edges allocation complete elapsed="
            << fixed << setprecision(1)
            << chrono::duration<double>(Clock::now() - up_alloc_start).count()
            << "s\n";

        auto down_alloc_start = Clock::now();
        cerr << "[CH-CSR] allocating reverse_downward.edges: "
            << reverse_downward_edges << " edges, approx "
            << fixed << setprecision(2)
            << (static_cast<long double>(reverse_downward_edges) * sizeof(QueryEdge)
                / (1024.0L * 1024.0L * 1024.0L))
            << " GiB\n";

        reverse_downward.allocate_edges(reverse_downward_edges);

        cerr << "[CH-CSR] reverse_downward.edges allocation complete elapsed="
            << fixed << setprecision(1)
            << chrono::duration<double>(Clock::now() - down_alloc_start).count()
            << "s\n";
    } catch (const bad_alloc&) {
        cerr << "[CH-CSR] ERROR: failed to allocate query graph storage\n";
        throw;
    }

    cerr << "[CH-CSR] all query graph edge storage allocated elapsed="
        << fixed << setprecision(1)
        << chrono::duration<double>(Clock::now() - alloc_start).count()
        << "s\n";

        run_query_graph_vertex_phase(
            "fill upward edges",
            threads,
            [&](int begin, int end) {
                for (int u = begin; u < end; ++u) {
                    uint64_t pos = upward.offset[u];
                    int ru = rank[u];

                    for (int eid : graph.out[u]) {
                        const Arc& e = graph.arcs[eid];
                        int rv = rank[e.to];

                        if (ru < 0 || rv < 0 || ru == rv) continue;

                        if (ru < rv) {
                            if (e.w > numeric_limits<EdgeWeight>::max()) {
                                throw runtime_error("query edge weight does not fit in EdgeWeight");
                            }

                            upward.edges[pos++] = QueryEdge{e.to, static_cast<EdgeWeight>(e.w)};
                        }
                    }
                }
            }
        );


        run_query_graph_vertex_phase(
            "fill reverse-downward edges",
            threads,
            [&](int begin, int end) {
                for (int v = begin; v < end; ++v) {
                    uint64_t pos = reverse_downward.offset[v];
                    int rv = rank[v];

                    for (int eid : graph.in[v]) {
                        const Arc& e = graph.arcs[eid];
                        int ru = rank[e.from];

                        if (ru < 0 || rv < 0 || ru == rv) continue;

                        if (ru > rv) {
                            if (e.w > numeric_limits<EdgeWeight>::max()) {
                                throw runtime_error("query edge weight does not fit in EdgeWeight");
                            }

                            reverse_downward.edges[pos++] =
                                QueryEdge{e.from, static_cast<EdgeWeight>(e.w)};
                        }
                    }
                }
            }
        );

        auto total_elapsed =
            chrono::duration<double>(Clock::now() - total_start).count();

        if (query_graph_progress_enabled) {
            cerr << "[CH-CSR] query graphs complete"
                << " upward_edges=" << upward.edge_count()
                << " reverse_downward_edges=" << reverse_downward.edge_count()
                << " total_edges="
                << (upward.edge_count() + reverse_downward.edge_count())
                << " elapsed=" << fixed << setprecision(1)
                << total_elapsed << "s"
                << "\n";
        }
    }

    static size_t run_dijkstra_all(
        int source,
        const CSRGraph& graph,
        vector<Weight>& dist,
        vector<int>& seen
    ) {
        using Item = pair<Weight, int>;
        priority_queue<Item, vector<Item>, greater<Item>> pq;

        dist[source] = 0;
        seen.push_back(source);
        pq.push({0, source});

        size_t settled = 0;

        while (!pq.empty()) {
            auto [d, u] = pq.top();
            pq.pop();

            if (d != dist[u]) continue;

            ++settled;

            uint64_t begin = graph.offset[u];
            uint64_t end = graph.offset[u + 1];

            for (uint64_t i = begin; i < end; ++i) {
                const QueryEdge& e = graph.edges[i];

                if (d > INF - e.w) continue;

                Weight nd = d + e.w;

                if (nd < dist[e.to]) {
                    if (dist[e.to] == INF) {
                        seen.push_back(e.to);
                    }

                    dist[e.to] = nd;
                    pq.push({nd, e.to});
                }
            }
        }

        return settled;
    }

    template <typename Body>
    void run_query_graph_vertex_phase(
        const string& phase_name,
        int threads,
        Body&& body
    ) const {
        using Clock = chrono::steady_clock;

        auto start = Clock::now();

        uint64_t total_vertices = static_cast<uint64_t>(n);
        uint64_t chunk_size = max<uint64_t>(1, query_graph_chunk_vertices);

        long long chunk_count =
            static_cast<long long>((total_vertices + chunk_size - 1) / chunk_size);

        atomic<uint64_t> completed_vertices{0};

        auto elapsed_seconds = [&]() -> double {
            return chrono::duration<double>(Clock::now() - start).count();
        };

        auto maybe_print_progress = [&](uint64_t previous, uint64_t current) {
            if (!query_graph_progress_enabled) return;

            bool crossed_progress_boundary =
                current == total_vertices ||
                current / query_graph_progress_every >
                    previous / query_graph_progress_every;

            if (!crossed_progress_boundary) return;

    #ifdef _OPENMP
            #pragma omp critical(ch_query_graph_progress_print)
    #endif
            {
                double elapsed = elapsed_seconds();
                double pct =
                    total_vertices == 0
                    ? 100.0
                    : 100.0 * double(current) / double(total_vertices);

                double rate =
                    elapsed > 0.0
                    ? double(current) / elapsed
                    : 0.0;

                cerr << "[CH-CSR] " << phase_name
                    << " progress "
                    << current << "/" << total_vertices
                    << " (" << fixed << setprecision(2) << pct << "%)"
                    << " elapsed=" << setprecision(1) << elapsed << "s"
                    << " rate=" << setprecision(0) << rate << " vertices/s"
                    << "\n";
            }
        };

        if (query_graph_progress_enabled) {
            cerr << "[CH-CSR] " << phase_name
                << " started"
                << " threads=" << threads
                << " chunk_vertices=" << chunk_size
                << "\n";
        }

    #ifdef _OPENMP
        if (threads > 1) {
            #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
            for (long long chunk = 0; chunk < chunk_count; ++chunk) {
                uint64_t begin64 = uint64_t(chunk) * chunk_size;
                uint64_t end64 = min<uint64_t>(begin64 + chunk_size, total_vertices);

                int begin = static_cast<int>(begin64);
                int end = static_cast<int>(end64);

                body(begin, end);

                uint64_t previous =
                    completed_vertices.fetch_add(end64 - begin64);

                uint64_t current = previous + (end64 - begin64);

                maybe_print_progress(previous, current);
            }
        } else
    #endif
        {
            for (long long chunk = 0; chunk < chunk_count; ++chunk) {
                uint64_t begin64 = uint64_t(chunk) * chunk_size;
                uint64_t end64 = min<uint64_t>(begin64 + chunk_size, total_vertices);

                int begin = static_cast<int>(begin64);
                int end = static_cast<int>(end64);

                body(begin, end);

                uint64_t previous =
                    completed_vertices.fetch_add(end64 - begin64);

                uint64_t current = previous + (end64 - begin64);

                maybe_print_progress(previous, current);
            }
        }

        if (query_graph_progress_enabled) {
            cerr << "[CH-CSR] " << phase_name
                << " complete"
                << " elapsed=" << fixed << setprecision(1)
                << elapsed_seconds() << "s"
                << "\n";
        }
    }

    void write_ch_text_impl(const string& path,
                            const CHTextWriteOptions& opts) const {
        using Clock = chrono::steady_clock;

        auto start = Clock::now();

        if (static_cast<int>(rank.size()) != n ||
            static_cast<int>(level.size()) != n) {
            throw runtime_error("cannot write CH: rank/level arrays have invalid size");
        }

        for (int v = 0; v < n; ++v) {
            if (rank[v] < 0) {
                throw runtime_error("cannot write CH: preprocessing is not complete");
            }
        }

        if (upward.n != n || reverse_downward.n != n) {
            throw runtime_error("cannot write CH: query graphs are not built");
        }

        vector<char> buffer(1 << 20);

        ofstream out;
        out.rdbuf()->pubsetbuf(buffer.data(), buffer.size());

        ios::openmode mode = ios::out;
        if (opts.binary_mode_newlines) {
            mode |= ios::binary;
        }

        out.open(path, mode);

        if (!out) {
            throw runtime_error("cannot open CH output file: " + path);
        }

        out.exceptions(ios::badbit | ios::failbit);

        auto elapsed_seconds = [&]() -> double {
            return chrono::duration<double>(Clock::now() - start).count();
        };

        auto print_progress = [&](const string& section,
                                uint64_t done,
                                uint64_t total) {
            double elapsed = elapsed_seconds();
            double pct = total == 0
                ? 100.0
                : 100.0 * double(done) / double(total);

            double rate = elapsed > 0.0
                ? double(done) / elapsed
                : 0.0;

            cerr << "[CH-WRITE] " << section
                << " " << done << "/" << total
                << " (" << fixed << setprecision(2) << pct << "%)"
                << " elapsed=" << setprecision(1) << elapsed << "s"
                << " rate=" << setprecision(0) << rate << " items/s"
                << "\n";
        };

        cerr << "[CH-WRITE] writing CH text file: " << path << "\n";

        out << "CH_TEXT_V2\n";
        out << "c vertex IDs are 1-based in this file\n";
        out << "c ranks are 0-based: lower rank means contracted earlier\n";
        out << "c no shortcut middle vertex is encoded as 0\n";

        out << "p ch"
            << " n " << n
            << " augmented_arcs " << graph.arcs.size()
            << " upward_edges " << upward.edge_count()
            << " downward_edges " << reverse_downward.edge_count()
            << "\n";

        out << "begin_vertices " << n << "\n";
        out << "c v <vertex_id> <rank> <level>\n";

        for (int v = 0; v < n; ++v) {
            out << "v "
                << (v + 1) << " "
                << rank[v] << " "
                << level[v] << "\n";

            uint64_t done = static_cast<uint64_t>(v) + 1;

            if (opts.progress_every > 0 &&
                (done == static_cast<uint64_t>(n) ||
                done % opts.progress_every == 0)) {
                print_progress("vertices", done, static_cast<uint64_t>(n));
            }
        }

        out << "end_vertices\n";

        if (opts.include_augmented_arcs) {
            uint64_t total_arcs = static_cast<uint64_t>(graph.arcs.size());

            out << "begin_augmented_arcs " << total_arcs << "\n";
            out << "c a <from> <to> <weight> <hops> <mid> <shortcut>\n";
            out << "c shortcut=0 means original input arc; shortcut=1 means CH shortcut\n";

            for (uint64_t i = 0; i < total_arcs; ++i) {
                const Arc& e = graph.arcs[static_cast<size_t>(i)];

                int mid_out = e.mid < 0 ? 0 : e.mid + 1;

                out << "a "
                    << (e.from + 1) << " "
                    << (e.to + 1) << " "
                    << e.w << " "
                    << e.hops << " "
                    << mid_out << " "
                    << (e.shortcut ? 1 : 0)
                    << "\n";

                uint64_t done = i + 1;

                if (opts.progress_every > 0 &&
                    (done == total_arcs || done % opts.progress_every == 0)) {
                    print_progress("augmented_arcs", done, total_arcs);
                }
            }

            out << "end_augmented_arcs\n";
        }

        auto write_csr_graph =
            [&](const string& name, const CSRGraph& g) {
                out << "begin_csr " << name
                    << " n " << g.n
                    << " edges " << g.edge_count()
                    << "\n";

                out << "c b <source_vertex> <degree>\n";
                out << "c e <target_vertex> <weight>\n";

                uint64_t written_edges = 0;
                uint64_t total_edges = g.edge_count();

                for (int v = 0; v < g.n; ++v) {
                    uint64_t begin = g.offset[v];
                    uint64_t end = g.offset[v + 1];
                    uint64_t deg = end - begin;

                    if (deg == 0) continue;

                    out << "b " << (v + 1) << " " << deg << "\n";

                    for (uint64_t i = begin; i < end; ++i) {
                        const QueryEdge& e = g.edges[i];

                        out << "e "
                            << (e.to + 1) << " "
                            << e.w << "\n";

                        ++written_edges;

                        if (opts.progress_every > 0 &&
                            (written_edges == total_edges ||
                            written_edges % opts.progress_every == 0)) {
                            print_progress(name, written_edges, total_edges);
                        }
                    }
                }

                out << "end_csr " << name << "\n";
            };

        auto write_downward_graph_from_augmented_graph =
            [&](const string& name) {
                uint64_t expected_edges = reverse_downward.edge_count();

                cerr << "[CH-WRITE] " << name
                    << " started, expected_edges=" << expected_edges
                    << "\n";

                out << "begin_csr " << name
                    << " n " << n
                    << " edges " << expected_edges
                    << "\n";

                out << "c This section prints downward arcs in their natural CH orientation.\n";
                out << "c b <source_vertex> <degree>\n";
                out << "c e <target_vertex> <weight>\n";
                out << "c Every printed edge u -> v satisfies rank(u) > rank(v).\n";
                out << "c Internally, CH queries may store these edges reversed as v -> u.\n";

                uint64_t written_edges = 0;

                for (int u = 0; u < n; ++u) {
                    int ru = rank[u];

                    uint64_t deg = 0;

                    for (int eid : graph.out[u]) {
                        const Arc& e = graph.arcs[eid];
                        int v = e.to;
                        int rv = rank[v];

                        if (ru < 0 || rv < 0 || ru == rv) continue;

                        if (ru > rv) {
                            ++deg;
                        }
                    }

                    if (deg != 0) {
                        out << "b " << (u + 1) << " " << deg << "\n";

                        for (int eid : graph.out[u]) {
                            const Arc& e = graph.arcs[eid];
                            int v = e.to;
                            int rv = rank[v];

                            if (ru < 0 || rv < 0 || ru == rv) continue;

                            if (ru > rv) {
                                out << "e "
                                    << (v + 1) << " "
                                    << e.w << "\n";

                                ++written_edges;

                                if (opts.progress_every > 0 &&
                                    (written_edges == expected_edges ||
                                    written_edges % opts.progress_every == 0)) {
                                    print_progress(name, written_edges, expected_edges);
                                }
                            }
                        }
                    }
                }

                out << "end_csr " << name << "\n";

                cerr << "[CH-WRITE] " << name
                    << " complete, written_edges=" << written_edges
                    << " expected_edges=" << expected_edges
                    << "\n";

                if (written_edges != expected_edges) {
                    throw runtime_error(
                        "downward writer edge count mismatch: wrote " +
                        to_string(written_edges) +
                        ", expected " +
                        to_string(expected_edges)
                    );
                }
            };

        write_csr_graph("upward", upward);

        cerr << "[CH-WRITE] upward section complete\n";

        write_downward_graph_from_augmented_graph("downward");

        out << "end_ch\n";

        out.close();

        cerr << "[CH-WRITE] complete"
            << " elapsed=" << fixed << setprecision(1)
            << elapsed_seconds() << "s"
            << "\n";
    }
};

struct RandomCheckSummary {
    size_t queries = 0;
    size_t matches = 0;
    size_t mismatches = 0;

    long double ch_total_ns = 0;
    long double dijkstra_total_ns = 0;

    size_t ch_settled_total = 0;
    size_t dijkstra_settled_total = 0;
};

RandomCheckSummary run_random_checks(
    const AdjList& original_adj,
    const ContractionHierarchy& ch,
    size_t query_count,
    uint64_t seed,
    size_t mismatch_print_limit = 10
) {
    using Clock = chrono::steady_clock;

    int n = ch.vertex_count();

    mt19937_64 rng(seed);
    uniform_int_distribution<int> vertex_dist(0, n - 1);

    CHQueryWorkspace ch_ws(n);
    DijkstraWorkspace dij_ws(n);

    RandomCheckSummary summary;
    summary.queries = query_count;

    size_t progress_every = max<size_t>(1, query_count / 10);

    cerr << "[CHECK] running " << query_count
         << " random CH-vs-Dijkstra queries, seed=" << seed << "\n";

    for (size_t i = 1; i <= query_count; ++i) {
        int s = vertex_dist(rng);
        int t = vertex_dist(rng);

        auto ch_start = Clock::now();
        CHQueryResult ch_result = ch.query_distance(s, t, ch_ws);
        auto ch_end = Clock::now();

        size_t dij_settled = 0;

        auto dij_start = Clock::now();
        Weight dij_dist = dij_ws.run(original_adj, s, t, &dij_settled);
        auto dij_end = Clock::now();

        auto ch_ns =
            chrono::duration_cast<chrono::nanoseconds>(ch_end - ch_start).count();

        auto dij_ns =
            chrono::duration_cast<chrono::nanoseconds>(dij_end - dij_start).count();

        summary.ch_total_ns += ch_ns;
        summary.dijkstra_total_ns += dij_ns;

        summary.ch_settled_total +=
            ch_result.settled_forward + ch_result.settled_reverse;

        summary.dijkstra_settled_total += dij_settled;

        if (same_distance(ch_result.distance, dij_dist)) {
            ++summary.matches;
        } else {
            ++summary.mismatches;

            if (summary.mismatches <= mismatch_print_limit) {
                cerr << "[CHECK] MISMATCH query #" << i
                     << " s=" << (s + 1)
                     << " t=" << (t + 1)
                     << " ch=" << format_distance(ch_result.distance)
                     << " dijkstra=" << format_distance(dij_dist)
                     << " ch_ns=" << ch_ns
                     << " dijkstra_ns=" << dij_ns
                     << "\n";
            }
        }

        if (i % progress_every == 0 || i == query_count) {
            cerr << "[CHECK] progress "
                 << i << "/" << query_count
                 << " mismatches=" << summary.mismatches << "\n";
        }
    }

    long double q = static_cast<long double>(query_count);

    cerr << "\n[CHECK] summary\n";
    cerr << "  queries:              " << summary.queries << "\n";
    cerr << "  matches:              " << summary.matches << "\n";
    cerr << "  mismatches:           " << summary.mismatches << "\n";
    cerr << "  avg CH query:         "
         << fixed << setprecision(3)
         << static_cast<double>(summary.ch_total_ns / q / 1000.0L)
         << " us\n";
    cerr << "  avg Dijkstra query:   "
         << fixed << setprecision(3)
         << static_cast<double>(summary.dijkstra_total_ns / q / 1000.0L)
         << " us\n";

    if (summary.ch_total_ns > 0) {
        cerr << "  Dijkstra / CH speedup:"
             << fixed << setprecision(2)
             << static_cast<double>(summary.dijkstra_total_ns / summary.ch_total_ns)
             << "x\n";
    }

    cerr << "  avg CH settled:       "
         << fixed << setprecision(1)
         << static_cast<double>(
                static_cast<long double>(summary.ch_settled_total) / q
            )
         << "\n";

    cerr << "  avg Dijkstra settled: "
         << fixed << setprecision(1)
         << static_cast<double>(
                static_cast<long double>(summary.dijkstra_settled_total) / q
            )
         << "\n\n";

    return summary;
}

struct ProgramOptions {
    string graph_path;
    size_t random_checks = 0;
    uint64_t seed = 1;
    int witness_threads = 1;
    size_t parallel_witness_min_pairs = 32;
    bool no_stdin = false;
    bool compare_stdin = false;
    int query_graph_threads = 1;
    bool query_graph_progress = true;
    uint64_t query_graph_progress_every = 1'000'000;
    uint64_t query_graph_chunk_vertices = 65'536;
    string write_ch_path;
    bool write_ch_include_augmented_arcs = true;
    uint64_t write_ch_progress_every = 5'000'000;
};

void print_usage(const char* program_name) {
    cerr << "usage: " << program_name << " graph.gr [options]\n\n"
         << "options:\n"
         << "  --check N       run N random CH-vs-Dijkstra correctness checks\n"
         << "  --seed S        random seed for --check, default 1\n"
         << "  --witness-threads T     number of threads for parallel witness searches, 0 = OpenMP max\n"
         << "  --parallel-min-pairs N  only parallelize evaluate(v) if it has at least N candidate checks\n"
         << "  --query-graph-threads T       threads for upward/downward graph construction, 0 = OpenMP max\n"
         << "  --query-graph-progress-every N print CSR build progress every N vertices\n"
         << "  --query-graph-chunk N         vertices per parallel CSR build chunk, default 65536\n"
         << "  --no-query-graph-progress     disable CSR graph build progress output\n"
         << "  --no-stdin      exit after preprocessing / random checks\n"
         << "  --compare-stdin              for stdin queries, also run regular Dijkstra and compare\n"
         << "  --write-ch PATH              write preprocessed CH graph to a text file\n"
         << "  --write-ch-query-only        skip augmented arc list from CH text output\n"
         << "  --write-ch-progress-every N  print CH write progress every N items\n"
         << "stdin query format:\n"
         << "  s t             1-based source and target vertices\n\n"
         << "examples:\n"
         << "  " << program_name << " graph.gr --check 1000 --no-stdin\n"
         << "  " << program_name << " graph.gr --check 100 --seed 42\n";
}

ProgramOptions parse_options(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        exit(1);
    }

    ProgramOptions opts;
    opts.graph_path = argv[1];

    for (int i = 2; i < argc; ++i) {
        string arg = argv[i];

        if (arg == "--check") {
            if (i + 1 >= argc) {
                throw runtime_error("--check requires a number");
            }

            opts.random_checks = stoull(argv[++i]);
        } else if (arg == "--seed") {
            if (i + 1 >= argc) {
                throw runtime_error("--seed requires a number");
            }

            opts.seed = stoull(argv[++i]);
        } else if (arg == "--no-stdin") {
            opts.no_stdin = true;
        } else if (arg == "--compare-stdin") {
            opts.compare_stdin = true;
        } else if (arg == "--witness-threads") {
            if (i + 1 >= argc) {
                throw runtime_error("--witness-threads requires a number");
            }

            opts.witness_threads = stoi(argv[++i]);
        } else if (arg == "--parallel-min-pairs") {
            if (i + 1 >= argc) {
                throw runtime_error("--parallel-min-pairs requires a number");
            }

            opts.parallel_witness_min_pairs = stoull(argv[++i]);
        } else if (arg == "--query-graph-threads") {
            if (i + 1 >= argc) {
                throw runtime_error("--query-graph-threads requires a number");
            }

            opts.query_graph_threads = stoi(argv[++i]);
        } else if (arg == "--query-graph-progress-every") {
            if (i + 1 >= argc) {
                throw runtime_error("--query-graph-progress-every requires a number");
            }

            opts.query_graph_progress_every = stoull(argv[++i]);
        } else if (arg == "--query-graph-chunk") {
            if (i + 1 >= argc) {
                throw runtime_error("--query-graph-chunk requires a number");
            }

            opts.query_graph_chunk_vertices = stoull(argv[++i]);
        } else if (arg == "--no-query-graph-progress") {
            opts.query_graph_progress = false;
        } else if (arg == "--write-ch") {
            if (i + 1 >= argc) {
                throw runtime_error("--write-ch requires a path");
            }

            opts.write_ch_path = argv[++i];
        } else if (arg == "--write-ch-query-only") {
            opts.write_ch_include_augmented_arcs = false;
        } else if (arg == "--write-ch-progress-every") {
            if (i + 1 >= argc) {
                throw runtime_error("--write-ch-progress-every requires a number");
            }

            opts.write_ch_progress_every = stoull(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
        } else {
            throw runtime_error("unknown option: " + arg);
        }
    }

    return opts;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    ProgramOptions opts;

    try {
        opts = parse_options(argc, argv);
    } catch (const exception& e) {
        cerr << "argument error: " << e.what() << "\n\n";
        print_usage(argv[0]);
        return 1;
    }

    ifstream fin(opts.graph_path);

    if (!fin) {
        cerr << "cannot open " << opts.graph_path << "\n";
        return 1;
    }

    DynGraph g = read_dimacs_gr(fin);

    cerr << "read n=" << g.n
         << " arcs=" << g.arcs.size()
         << "\n";

    // Dijkstra-searchable copy of the original graph
    AdjList original_adj = build_original_adjacency(g);

    ContractionHierarchy ch(std::move(g));
    ch.set_witness_threads(opts.witness_threads);
    ch.set_parallel_witness_min_pairs(opts.parallel_witness_min_pairs);
    ch.set_query_graph_threads(opts.query_graph_threads);
    ch.set_query_graph_progress(
        opts.query_graph_progress,
        opts.query_graph_progress_every,
        opts.query_graph_chunk_vertices
    );

    cerr << "query graph construction threads: "
        << ch.get_query_graph_threads()
        << "\n";

    #ifdef _OPENMP

    //debug
    cerr << "OpenMP enabled, _OPENMP=" << _OPENMP
        << ", omp_get_max_threads()=" << omp_get_max_threads()
        << "\n";
    #else
    cerr << "OpenMP NOT enabled at compile time; witness searches will be serial.\n";
    #endif

    cerr << "parallel witness threads: "
        << ch.get_witness_threads()
        << "\n";

    auto preprocess_start = chrono::steady_clock::now();

    ch.preprocess();

    auto preprocess_end = chrono::steady_clock::now();

    if (!opts.write_ch_path.empty()) {
        CHTextWriteOptions write_opts;
        write_opts.include_augmented_arcs = opts.write_ch_include_augmented_arcs;
        write_opts.progress_every = opts.write_ch_progress_every;

        ch.write_ch_text(opts.write_ch_path, write_opts);
    }

    auto preprocess_ms =
        chrono::duration_cast<chrono::milliseconds>(
            preprocess_end - preprocess_start
        ).count();

    cerr << "preprocessing done in "
         << fixed << setprecision(3)
         << (preprocess_ms / 1000.0)
         << "s\n";

    size_t random_mismatches = 0;

    if (opts.random_checks > 0) {
        RandomCheckSummary summary =
            run_random_checks(original_adj, ch, opts.random_checks, opts.seed);

        random_mismatches = summary.mismatches;
    }

    if (opts.no_stdin) {
        return random_mismatches == 0 ? 0 : 2;
    }

    if (!opts.no_stdin) {
        cerr << "enter 1-based queries: s t\n";

        if (opts.compare_stdin) {
            cerr << "stdin mode: CH query + regular Dijkstra comparison\n";
        } else {
            cerr << "stdin mode: CH query only; use --compare-stdin to also run regular Dijkstra\n";
        }

        CHQueryWorkspace ch_ws(ch.vertex_count());
        DijkstraWorkspace dij_ws(ch.vertex_count());

        int s_in;
        int t_in;

        while (cin >> s_in >> t_in) {
            int s = s_in - 1;
            int t = t_in - 1;

            if (s < 0 || t < 0 || s >= ch.vertex_count() || t >= ch.vertex_count()) {
                cout << "ERROR invalid query vertices: "
                    << s_in << " " << t_in << "\n"
                    << flush;
                continue;
            }

            using Clock = chrono::steady_clock;

            auto ch_start = Clock::now();
            CHQueryResult ch_result = ch.query_distance(s, t, ch_ws);
            auto ch_end = Clock::now();

            long double ch_us =
                chrono::duration<long double, micro>(ch_end - ch_start).count();

            if (!opts.compare_stdin) {
                cout << format_distance(ch_result.distance)
                    << "\n"
                    << flush;

                continue;
            }

            size_t dijkstra_settled = 0;

            auto dij_start = Clock::now();
            Weight dij_dist =
                dij_ws.run(original_adj, s, t, &dijkstra_settled);
            auto dij_end = Clock::now();

            long double dij_us =
                chrono::duration<long double, micro>(dij_end - dij_start).count();

            bool ok = same_distance(ch_result.distance, dij_dist);

            cout << "s=" << s_in
                << " t=" << t_in
                << " ch=" << format_distance(ch_result.distance)
                << " dijkstra=" << format_distance(dij_dist)
                << " match=" << (ok ? "yes" : "NO")
                << " ch_us=" << fixed << setprecision(3) << static_cast<double>(ch_us)
                << " dijkstra_us=" << fixed << setprecision(3) << static_cast<double>(dij_us);

            if (ch_us > 0.0L) {
                cout << " speedup=" << fixed << setprecision(2)
                    << static_cast<double>(dij_us / ch_us) << "x";
            }

            cout << " ch_settled="
                << (ch_result.settled_forward + ch_result.settled_reverse)
                << " dijkstra_settled=" << dijkstra_settled
                << "\n"
                << flush;
        }
    }

    return random_mismatches == 0 ? 0 : 2;
}
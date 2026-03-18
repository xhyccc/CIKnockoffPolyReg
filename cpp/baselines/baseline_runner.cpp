/**
 * @file baseline_runner.cpp
 * @brief C++ comparison runner: loads a CSV dataset and runs all three
 *        C++ polynomial regression baselines (PolyLasso, PolyOMP, PolySTLSQ).
 *
 * Usage:
 * @code
 *   ./baseline_runner <data.csv> [degree] [output_prefix]
 * @endcode
 *
 * The CSV must have a header row; the last column is treated as the response.
 * Results are written to:
 *   <output_prefix>_results.json
 *   <output_prefix>_results.csv
 *
 * If no output_prefix is given, results are only printed to stdout.
 */

#include "baselines.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ic_knockoff;
using namespace ic_knockoff::baselines;

// ---------------------------------------------------------------------------
// CSV loader
// ---------------------------------------------------------------------------

struct CSVData {
    Matrix X;
    Vec y;
    std::vector<std::string> feature_names;
    std::string source;
};

static CSVData load_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open CSV: " + path);

    std::string line;
    bool first = true;
    std::vector<std::string> header;
    std::vector<std::vector<double>> rows;

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> fields;
        while (std::getline(ss, token, ','))
            fields.push_back(token);

        if (first) {
            // Try to parse as numbers; if it fails, treat as header
            bool is_header = false;
            for (const auto& t : fields) {
                try { std::stod(t); } catch (...) { is_header = true; break; }
            }
            if (is_header) {
                header = fields;
                first = false;
                continue;
            }
            // No header — auto-generate names
            for (std::size_t j = 0; j < fields.size(); ++j)
                header.push_back("x" + std::to_string(j));
            first = false;
        }

        std::vector<double> row;
        for (const auto& t : fields) {
            try { row.push_back(std::stod(t)); }
            catch (...) { row.push_back(0.0); }
        }
        if (!row.empty()) rows.push_back(row);
    }

    if (rows.empty())
        throw std::runtime_error("CSV has no data rows: " + path);

    std::size_t n = rows.size();
    std::size_t n_cols = rows[0].size();
    std::size_t p = n_cols - 1;  // last column is y

    Matrix X(n, p, 0.0);
    Vec y(n);
    std::vector<std::string> feat_names;
    for (std::size_t j = 0; j < p; ++j)
        feat_names.push_back(j < header.size() ? header[j] : "x" + std::to_string(j));

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j)
            X(i, j) = (j < rows[i].size()) ? rows[i][j] : 0.0;
        y[i] = (p < rows[i].size()) ? rows[i][p] : 0.0;
    }

    return {X, y, feat_names, path};
}

// ---------------------------------------------------------------------------
// JSON / CSV output helpers
// ---------------------------------------------------------------------------

static void write_json(const std::vector<ResultBundle>& results,
                        const std::string& path) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Warning: cannot write " << path << "\n";
        return;
    }
    f << "[\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (i) f << ",\n";
        f << results[i].to_json();
    }
    f << "\n]\n";
}

static void write_csv(const std::vector<ResultBundle>& results,
                       const std::string& path) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Warning: cannot write " << path << "\n";
        return;
    }
    for (std::size_t i = 0; i < results.size(); ++i)
        f << results[i].to_csv_row(i == 0);
}

static void print_table(const std::vector<ResultBundle>& results) {
    std::cout << "\n";
    std::printf("%-22s  %8s  %8s  %10s  %10s\n",
                "Method", "N_sel", "R2", "BIC", "Elapsed(s)");
    std::cout << std::string(66, '-') << "\n";
    for (const auto& r : results) {
        auto fmt = [](double v) -> std::string {
            if (std::isnan(v) || std::isinf(v)) return "    N/A";
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%8.4f", v);
            return buf;
        };
        std::printf("%-22s  %8d  %s  %s  %s\n",
                    r.method.c_str(), r.n_selected,
                    fmt(r.r_squared).c_str(),
                    fmt(r.bic).c_str(),
                    fmt(r.elapsed_seconds).c_str());
    }
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <data.csv> [degree=2] [output_prefix]\n";
        return 1;
    }

    std::string csv_path = argv[1];
    int degree = (argc >= 3) ? std::atoi(argv[2]) : 2;
    std::string out_prefix = (argc >= 4) ? std::string(argv[3]) : "";

    CSVData data;
    try {
        data = load_csv(csv_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading CSV: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Dataset : " << csv_path << "\n"
              << "Shape   : " << data.X.rows << " x " << data.X.cols << "\n"
              << "Degree  : " << degree << "\n\n";

    std::vector<ResultBundle> results;

    // Helper lambda to time and run a baseline
    auto run = [&](const char* name, auto fit_fn) {
        std::cout << "Running " << name << " ...\n" << std::flush;
        auto t0 = std::chrono::steady_clock::now();
        auto [rb, msg] = fit_fn();
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        rb.elapsed_seconds = elapsed;
        results.push_back(rb);
        std::cout << "  selected=" << rb.n_selected
                  << "  R2=" << rb.r_squared
                  << "  t=" << elapsed << "s\n";
    };

    // ---- PolyLasso ----
    run("PolyLasso", [&]() -> std::pair<ResultBundle, std::string> {
        PolyLasso lasso;
        lasso.degree = degree;
        lasso.fit(data.X, data.y);
        auto exp = polynomial_expand(data.X, degree, true, 1e-8, {});
        return {lasso.to_result_bundle(data.X, data.y, exp, csv_path), ""};
    });

    // ---- PolyOMP ----
    run("PolyOMP", [&]() -> std::pair<ResultBundle, std::string> {
        PolyOMP omp;
        omp.degree = degree;
        omp.fit(data.X, data.y);
        auto exp = polynomial_expand(data.X, degree, true, 1e-8, {});
        return {omp.to_result_bundle(data.X, data.y, exp, csv_path), ""};
    });

    // ---- PolySTLSQ ----
    run("PolySTLSQ", [&]() -> std::pair<ResultBundle, std::string> {
        PolySTLSQ stlsq;
        stlsq.degree = degree;
        stlsq.fit(data.X, data.y);
        auto exp = polynomial_expand(data.X, degree, true, 1e-8, {});
        return {stlsq.to_result_bundle(data.X, data.y, exp, csv_path), ""};
    });

    // ---- Output ----
    print_table(results);

    if (!out_prefix.empty()) {
        write_json(results, out_prefix + "_results.json");
        write_csv(results, out_prefix + "_results.csv");
        std::cout << "Results written to " << out_prefix << "_results.json/.csv\n";
    }

    return 0;
}

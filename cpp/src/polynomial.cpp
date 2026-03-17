#include "ic_knockoff/polynomial.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace ic_knockoff {

std::size_t n_expanded_features(std::size_t n_base, int degree, bool include_bias) {
    return n_base * 2 * static_cast<std::size_t>(degree) + (include_bias ? 1 : 0);
}

ExpandedFeatures polynomial_expand(
    const Matrix& X,
    int degree,
    bool include_bias,
    double clip_threshold,
    const std::vector<std::string>& base_names) {

    if (degree < 1)
        throw std::invalid_argument("polynomial_expand: degree must be >= 1");

    std::size_t n = X.rows;
    std::size_t p = X.cols;

    // Validate base_names
    std::vector<std::string> names;
    if (base_names.empty()) {
        for (std::size_t j = 0; j < p; ++j)
            names.push_back("x" + std::to_string(j));
    } else {
        if (base_names.size() != p)
            throw std::invalid_argument(
                "polynomial_expand: base_names.size() must equal X.cols");
        names = base_names;
    }

    std::size_t n_out = n_expanded_features(p, degree, include_bias);
    Matrix out(n, n_out, 0.0);
    std::vector<ExpandedFeatureInfo> info;
    info.reserve(n_out);

    std::size_t col = 0;
    for (std::size_t j = 0; j < p; ++j) {
        // Positive powers: d = 1 to degree
        for (int d = 1; d <= degree; ++d) {
            for (std::size_t i = 0; i < n; ++i) {
                double xij = X(i, j);
                out(i, col) = std::pow(xij, d);
            }
            std::string feat_name = (d == 1)
                ? names[j]
                : (names[j] + "^" + std::to_string(d));
            info.push_back({static_cast<int>(j), d, feat_name});
            ++col;
        }
        // Negative powers: d = 1 to degree
        for (int d = 1; d <= degree; ++d) {
            for (std::size_t i = 0; i < n; ++i) {
                double xij = X(i, j);
                // Clip near-zero values
                if (std::abs(xij) < clip_threshold) {
                    xij = (xij >= 0.0) ? clip_threshold : -clip_threshold;
                }
                out(i, col) = std::pow(xij, -d);
            }
            std::string feat_name = names[j] + "^(-" + std::to_string(d) + ")";
            info.push_back({static_cast<int>(j), -d, feat_name});
            ++col;
        }
    }

    // Optional bias column
    if (include_bias) {
        for (std::size_t i = 0; i < n; ++i)
            out(i, col) = 1.0;
        info.push_back({-1, 0, "1"});
        ++col;
    }

    return ExpandedFeatures{out, info};
}

} // namespace ic_knockoff

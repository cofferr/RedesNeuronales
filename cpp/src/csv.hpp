#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace csv {

inline std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) fields.push_back(field);
    return fields;
}

inline std::ifstream open(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    return in;
}

// Reads a numeric CSV into a matrix. skip_header drops the first line.
inline Eigen::MatrixXd read_matrix(const std::string& path, bool skip_header) {
    std::ifstream in = open(path);
    std::string line;
    if (skip_header) std::getline(in, line);
    std::vector<std::vector<double>> rows;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<double> row;
        for (const auto& f : split(line)) row.push_back(std::stod(f));
        if (!rows.empty() && row.size() != rows[0].size())
            throw std::runtime_error("ragged rows in " + path);
        rows.push_back(std::move(row));
    }
    if (rows.empty()) throw std::runtime_error("no data in " + path);
    Eigen::MatrixXd m(rows.size(), rows[0].size());
    for (std::size_t i = 0; i < rows.size(); ++i)
        for (std::size_t j = 0; j < rows[i].size(); ++j) m(i, j) = rows[i][j];
    return m;
}

// Reads a "name,value" CSV (with header) into a map.
inline std::map<std::string, double> read_named_values(const std::string& path) {
    std::ifstream in = open(path);
    std::string line;
    std::getline(in, line);
    std::map<std::string, double> values;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto f = split(line);
        if (f.size() != 2) throw std::runtime_error("bad line in " + path + ": " + line);
        values[f[0]] = std::stod(f[1]);
    }
    return values;
}

}  // namespace csv

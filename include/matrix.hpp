// matrix.hpp
#pragma once            // include this header at most once per translation unit
#include <cstddef>      // std::size_t
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

class Matrix {
public:
    using size_type = std::size_t;   // an alias, like a type hint you can reuse

    Matrix() = default;              // empty 0x0 matrix

    // Matrix m(2, 3);       -> np.zeros((2, 3))
    // Matrix m(2, 3, 1.0);  -> np.full((2, 3), 1.0)
    Matrix(size_type rows, size_type cols, double fill = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    // Matrix m{{1, 2}, {3, 4}};  -> np.array([[1, 2], [3, 4]])
    Matrix(std::initializer_list<std::initializer_list<double>> init)
        : rows_(init.size()),
          cols_(init.size() ? init.begin()->size() : 0) {
        data_.reserve(rows_ * cols_);
        for (const auto& row : init) {
            if (row.size() != cols_)
                throw std::invalid_argument("Matrix: ragged initializer list");
            data_.insert(data_.end(), row.begin(), row.end());
        }
    }

    size_type rows() const noexcept { return rows_; }
    size_type cols() const noexcept { return cols_; }

private:
    size_type rows_ = 0;
    size_type cols_ = 0;
    std::vector<double> data_;   // row-major: (i, j) -> i * cols_ + j
};

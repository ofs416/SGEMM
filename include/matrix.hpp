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

    // const indicates this returns a copy and is not able to write
    size_type rows() const noexcept { return rows_; }
    size_type cols() const noexcept { return cols_; }

    // Two versions to allow for read only (const) Matrix
    double&       operator()(size_type i, size_type j)       { return data_[i * cols_ + j]; }
    const double& operator()(size_type i, size_type j) const { return data_[i * cols_ + j]; }

    // Checked versions, like std::vector::at
    double& at(size_type i, size_type j) {
        check_index(i, j);
        return (*this)(i, j);
    }
    const double& at(size_type i, size_type j) const {
        check_index(i, j);
        return (*this)(i, j);
    }

    // Addition
    Matrix& operator+=(const Matrix& rhs) {
        check_same_shape(rhs, "+=");
        for (size_type k = 0; k < data_.size(); ++k)
            data_[k] += rhs.data_[k];
        return *this;
    }

    // Subtraction
    Matrix& operator-=(const Matrix& rhs) {
        check_same_shape(rhs, "-=");
        for (size_type k = 0; k < data_.size(); ++k)
            data_[k] -= rhs.data_[k];
        return *this;
    }

    // Scalar multiplication
    Matrix& operator*=(double s) {
        for (auto& x : data_) x *= s;
        return *this;
    }

    // Element wise multiplication
    Matrix& operator*=(const Matrix& rhs) {        // function overloads
        check_same_shape(rhs, "*=");
        for (size_type k = 0; k < data_.size(); ++k)
            data_[k] *= rhs.data_[k];
        return *this;
    }

private:
    size_type rows_ = 0;
    size_type cols_ = 0;
    std::vector<double> data_;   // row-major: (i, j) -> i * cols_ + j

    void check_index(size_type i, size_type j) const {
        if (i >= rows_ || j >= cols_)
            throw std::out_of_range("Matrix: index out of range");
    }

    void check_same_shape(const Matrix& rhs, const char* op) const {
        if (rows_ != rhs.rows() || cols_ != rhs.cols())      // .rows() instead of .rows_
            throw std::invalid_argument(std::string("Matrix ") + op + ": shape mismatch");
    }
};

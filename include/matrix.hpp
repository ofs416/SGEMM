// matrix.hpp
#pragma once            // include this header at most once per translation unit
#include <cstddef>      // std::size_t
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

// T is the element type, like dtype in numpy: BasicMatrix<float> -> dtype=np.float32
template <typename T>
class BasicMatrix {
public:
    using value_type = T;
    using size_type = std::size_t;   // an alias, like a type hint you can reuse

    BasicMatrix() = default;         // empty 0x0 matrix

    // BasicMatrix<T> m(2, 3);       -> np.zeros((2, 3))   (T{} is 0 for numbers)
    // BasicMatrix<T> m(2, 3, 1.0);  -> np.full((2, 3), 1.0)
    BasicMatrix(size_type rows, size_type cols, T fill = T{})
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    // BasicMatrix<T> m{{1, 2}, {3, 4}};  -> np.array([[1, 2], [3, 4]])
    BasicMatrix(std::initializer_list<std::initializer_list<T>> init)
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
    T&       operator()(size_type i, size_type j)       { return data_[i * cols_ + j]; }
    const T& operator()(size_type i, size_type j) const { return data_[i * cols_ + j]; }

    // Checked versions, like std::vector::at
    T& at(size_type i, size_type j) {
        check_index(i, j);
        return (*this)(i, j);
    }
    const T& at(size_type i, size_type j) const {
        check_index(i, j);
        return (*this)(i, j);
    }

    // Inside the template, plain BasicMatrix means BasicMatrix<T>

    // Matrix Addition
    BasicMatrix& operator+=(const BasicMatrix& rhs) {
        check_same_shape(rhs, "+=");
        for (size_type k = 0; k < data_.size(); ++k)
            data_[k] += rhs.data_[k];
        return *this;
    }
    friend BasicMatrix operator+(BasicMatrix lhs, const BasicMatrix& rhs) {
        lhs += rhs;
        return lhs;
    }

    // Scalar Addition
    BasicMatrix& operator+=(const T s) {
        for (auto& x : data_) x += s;
        return *this;
    }
    friend BasicMatrix operator+(BasicMatrix lhs, const T s) {
        lhs += s;
        return lhs;
    }
    friend BasicMatrix operator+(const T s, BasicMatrix rhs) {
        rhs += s;
        return rhs;
    }

    // Matrix Subtraction
    BasicMatrix& operator-=(const BasicMatrix& rhs) {
        check_same_shape(rhs, "-=");
        for (size_type k = 0; k < data_.size(); ++k)
            data_[k] -= rhs.data_[k];
        return *this;
    }
    friend BasicMatrix operator-(BasicMatrix lhs, const BasicMatrix& rhs) {
        lhs -= rhs;
        return lhs;
    }

    // Scalar Subtraction
    BasicMatrix& operator-=(const T s) {
        for (auto& x : data_) x -= s;
        return *this;
    }
    friend BasicMatrix operator-(BasicMatrix lhs, const T s) {
        lhs -= s;
        return lhs;
    }
    friend BasicMatrix operator-(const T s, BasicMatrix rhs) {
        for (auto& x : rhs.data_) x = s - x;
        return rhs;
    }

    // Scalar multiplication
    BasicMatrix& operator*=(const T s) {
        for (auto& x : data_) x *= s;
        return *this;
    }
    friend BasicMatrix operator*(BasicMatrix lhs, const T s) {
        lhs *= s;
        return lhs;
    }
    friend BasicMatrix operator*(const T s, BasicMatrix rhs) {
        rhs *= s;
        return rhs;
    }
    friend BasicMatrix operator-(BasicMatrix m) {   // friend version avoids one extra copy
        for (auto& x : m.data_) x = -x;             // negate directly, no -1.0 literal needed
        return m;
    }

    // Element-wise multiplication
    BasicMatrix& operator*=(const BasicMatrix& rhs) {   // function overloads
        check_same_shape(rhs, "*=");
        for (size_type k = 0; k < data_.size(); ++k)
            data_[k] *= rhs.data_[k];
        return *this;
    }
    friend BasicMatrix operator*(BasicMatrix lhs, const BasicMatrix& rhs) {
        lhs *= rhs;
        return lhs;
    }

private:
    size_type rows_ = 0;
    size_type cols_ = 0;
    std::vector<T> data_;   // row-major: (i, j) -> i * cols_ + j

    void check_index(size_type i, size_type j) const {
        if (i >= rows_ || j >= cols_)
            throw std::out_of_range("Matrix: index out of range");
    }

    void check_same_shape(const BasicMatrix& rhs, const char* op) const {
        if (rows_ != rhs.rows_ || cols_ != rhs.cols_)
            throw std::invalid_argument(std::string("Matrix ") + op + ": shape mismatch");
    }
};

// Like std::string = std::basic_string<char>
using Matrix  = BasicMatrix<double>;
using MatrixF = BasicMatrix<float>;   // single precision: the "S" in SGEMM

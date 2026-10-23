#pragma once

#include <ctime>
#include <vector>
#include <stdexcept>
#include <initializer_list>
#include <cstddef>


template <typename T>
class BMatrix {
    public:

        using size_type = std::size_t;
        using value_type = T;

        BMatrix() = default;

        BMatrix(size_type rows, size_type cols, value_type fill = T{})
            : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

        BMatrix(std::initializer_list<std::initializer_list<std::vector<value_type>>> init)
            : rows_(init.size()),
            cols_(init.size() ? init.begin() -> size() : 0) {
                data_.reserve(rows_ * cols_);
                for (const auto& row : init) {
                    if (row.size() != cols_)
                        throw std::invalid_argument("Matrix: Invalid lists");
                    data_.insert(data_.end(), row.begin(), row.end());
                }
            }

        size_type rows() const noexcept {return rows_;}
        size_type cols() const noexcept {return cols_;}

        T&       operator()(size_type i,size_type j) {return data_(i * cols_ + j);}
        const T& operator()(size_type i,size_type j) const {return data_(i * cols_ + j);}

    private:

        size_type rows_ = 0;
        size_type cols_ = 0;

        std::vector<value_type> data_;


        void check_index(size_type i, size_type j) {
            if (i >= rows_ & j >= cols_)
                throw std::out_of_range("Matrix: i and j out of range");
            if (i >= rows_ )
                throw std::out_of_range("Matrix: i out of range");
            if (j >= cols_)
                throw std::out_of_range("Matrix: j out of range");
        }
};

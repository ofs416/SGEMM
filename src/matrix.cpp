#include <vector>
#include <cstddef>
#include <random>

class Matrix {
public:
    Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {}

    float& operator()(std::size_t i, std::size_t j)       { return data_[i * cols_ + j]; }
    float  operator()(std::size_t i, std::size_t j) const { return data_[i * cols_ + j]; }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    float* data() { return data_.data(); }

private:
    std::size_t rows_, cols_;
    std::vector<float> data_;
};

int main() {

    return 0;
}

class Matrix {
public:
    static constexpr std::size_t kAlign       = 64;                    // bytes
    static constexpr std::size_t kAlignFloats = kAlign / sizeof(float); // 16
    // ...
private:
    struct AlignedDelete {
        void operator()(float* p) const noexcept {
            ::operator delete[](p, std::align_val_t{kAlign});
        }
    };
    using Buffer = std::unique_ptr<float[], AlignedDelete>;

    static Buffer allocate(std::size_t n) {
        if (n == 0) return Buffer{};
        void* p = ::operator new[](n * sizeof(float), std::align_val_t{kAlign});
        return Buffer{static_cast<float*>(p)};
    }

    static constexpr std::size_t round_up(std::size_t x, std::size_t m) noexcept {
        return (x + m - 1) / m * m;
    }

    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::size_t ld_   = 0;   // floats between the start of row i and row i+1
    Buffer      data_;
};

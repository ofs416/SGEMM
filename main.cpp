#include <iostream>
#include "include/sgemm.hpp"

int main() {
    MatrixF a(3,5, 2.0);
    MatrixF b(5,3, 3.0);

    MatrixF c = matmul(a, b);

    std::cout << c(1,1) << "\n";
}

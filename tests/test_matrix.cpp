#include <cassert>
#include <iostream>
#include <stdexcept>
#include "../include/matrix.hpp"

int main() {
    Matrix empty;
    assert(empty.rows() == 0 && empty.cols() == 0);
    std::cout << "empty:   " << empty.rows() << "x" << empty.cols() << "\n";

    Matrix zeros(3, 4);
    assert(zeros.rows() == 3 && zeros.cols() == 4);
    std::cout << "zeros:   " << zeros.rows() << "x" << zeros.cols() << "\n";

    Matrix sevens(2, 2, 7.0);
    assert(sevens.rows() == 2 && sevens.cols() == 2);
    std::cout << "sevens:  " << sevens.rows() << "x" << sevens.cols() << "\n";

    Matrix m{{1, 2, 3},
             {4, 5, 6}};
    assert(m.rows() == 2 && m.cols() == 3);
    std::cout << "m:       " << m.rows() << "x" << m.cols() << "\n";

    assert(m(0, 0) == 1 && m(0,1) == 2);
    std::cout << "m(0, 0) == " << m(0, 0) << " and " << "m(0, 1) == " << m(0, 1) << "\n";

    try {
        Matrix bad{{1, 2}, {3}};
        std::cerr << "FAIL: expected exception\n";
        return 1;
    } catch (const std::invalid_argument& e) {
        std::cout << "ragged:  caught \"" << e.what() << "\"\n";
    }

    try {
        m.at(2,0);
        std::cerr << "FAIL: expected exception\n";
        return 1;
    } catch (const std::out_of_range& e) {
        std::cout << "Out of bounds:  caught \"" << e.what() << "\"\n";
    }


    std::cout << "all assertions passed\n";
}

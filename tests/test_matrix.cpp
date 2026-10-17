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

    Matrix twos(2, 2, 2.0);
    Matrix nines(2, 2, 9.0);
    twos += sevens;
    for (int i = 0; i < twos.rows(); i++) {
        for (int j = 0; j < twos.cols(); j++) {
            assert(twos(i, j) == nines(i, j));
        };
    };
    std::cout << "twos += sevens == nines \n";

    twos -= sevens;
    for (int i = 0; i < twos.rows(); i++) {
        for (int j = 0; j < twos.cols(); j++) {
            assert(twos(i, j) == 2.0);
        };
    };
    std::cout << "twos is 2.0 again! \n";

    twos *= 2.0;
    for (int i = 0; i < twos.rows(); i++) {
        for (int j = 0; j < twos.cols(); j++) {
            assert(twos(i, j) == 4.0);
        };
    };
    std::cout << "twos *= 2.0 gives 4.0 \n";

    Matrix a{{1, 2, 3},
             {4, 5, 6}};
    Matrix b{{2, 0, -1},
             {0.5, 3, 10}};
    Matrix ab{{2, 0, -3},
              {2, 15, 60}};
    a *= b;
    for (int i = 0; i < a.rows(); i++) {
        for (int j = 0; j < a.cols(); j++) {
            assert(a(i, j) == ab(i, j));
        };
    };
    std::cout << "a *= b is element wise \n";

    try {
        a *= twos;
        std::cerr << "FAIL: expected exception\n";
        return 1;
    } catch (const std::invalid_argument& e) {
        std::cout << "*= shape mismatch:  caught \"" << e.what() << "\"\n";
    }
    for (int i = 0; i < a.rows(); i++) {
        for (int j = 0; j < a.cols(); j++) {
            assert(a(i, j) == ab(i, j));
        };
    };
    std::cout << "a unchanged after failed *= \n";

    Matrix fours(2, 2, 4.0);
    {
    Matrix eights = fours * 2;
    assert(eights(0, 0) == 8.0);
    std::cout << "fours * 2 == eights\n";
    }
    {
    Matrix eights = 2 * fours;
    assert(eights(0, 0) == 8.0);
    std::cout << "2 * fours  == eights\n";
    }

    std::cout << "all assertions passed\n";
}

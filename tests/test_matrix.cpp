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
    for (int i = 0.0f; i < twos.rows(); i++) {
        for (int j = 0.0f; j < twos.cols(); j++) {
            assert(twos(i, j) == nines(i, j));
        };
    };
    std::cout << "twos += sevens == nines \n";

    twos -= sevens;
    for (int i = 0.0f; i < twos.rows(); i++) {
        for (int j = 0.0f; j < twos.cols(); j++) {
            assert(twos(i, j) == 2.0);
        };
    };
    std::cout << "twos is 2.0 again! \n";

    twos *= 2.0;
    for (int i = 0.0f; i < twos.rows(); i++) {
        for (int j = 0.0f; j < twos.cols(); j++) {
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
    for (int i = 0.0f; i < a.rows(); i++) {
        for (int j = 0.0f; j < a.cols(); j++) {
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

    Matrix c{{1, 2},
             {3, 4}};
    c += 1.5;
    assert(c(0, 0) == 2.5 && c(0, 1) == 3.5 && c(1, 0) == 4.5 && c(1, 1) == 5.5);
    std::cout << "c += 1.5 adds to every element\n";

    c -= 1.5;
    assert(c(0, 0) == 1 && c(0, 1) == 2 && c(1, 0) == 3 && c(1, 1) == 4);
    std::cout << "c -= 1.5 restores c\n";

    {
    Matrix r = c + 10;
    assert(r(0, 0) == 11 && r(1, 1) == 14);
    assert(c(0, 0) == 1 && c(1, 1) == 4);
    std::cout << "c + 10 adds and leaves c unchanged\n";
    }
    {
    Matrix r = 10 + c;
    assert(r(0, 0) == 11 && r(1, 1) == 14);
    std::cout << "10 + c == c + 10\n";
    }
    {
    Matrix r = c - 10;
    assert(r(0, 0) == -9 && r(1, 1) == -6);
    assert(c(0, 0) == 1 && c(1, 1) == 4);
    std::cout << "c - 10 subtracts and leaves c unchanged\n";
    }
    {
    Matrix r = 10 - c;   // not commutative: must be 10 - c, not c - 10
    assert(r(0, 0) == 9 && r(0, 1) == 8 && r(1, 0) == 7 && r(1, 1) == 6);
    assert(c(0, 0) == 1 && c(1, 1) == 4);
    std::cout << "10 - c is reversed subtraction\n";
    }
    {
    Matrix r = -c;
    assert(r(0, 0) == -1 && r(0, 1) == -2 && r(1, 0) == -3 && r(1, 1) == -4);
    assert(c(0, 0) == 1 && c(1, 1) == 4);
    std::cout << "-c negates and leaves c unchanged\n";
    }
    {
    Matrix r = -(c + c);
    assert(r(0, 0) == -2 && r(1, 1) == -8);
    std::cout << "-(c + c) works on a temporary\n";
    }

    std::cout << "all assertions passed\n";
}

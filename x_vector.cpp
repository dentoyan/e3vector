
#include "e3vector.h"
#include <iostream>
#include <cmath>


int main()
{
    E3Vector v1 = { 1.0, 0.0, 2.0 };
    E3Vector v2 = { 2.0, 2.0, 2.0 };

    double a = acos((v1 * v2) / (v1.length() * v2.length()));

    std::cout << "alpha= " << a << " rad" << std::endl;
    return 0;
}


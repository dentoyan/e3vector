# E3Vector #


## About ##

Pure lean and fast vector arithmetic using AVX2 vector unit.
Use this Vector Class for numerical linear algebra.

Copyright (c) 2014-2018 Joshua Dentoyan


## Overview ##

The Vector math library provides 3-D vector operations including
addition, subtraction, dot product, multiply by a const etc.


## Build ##

g++ -mavx2 -o x_vector x_vector.cpp

## Usage ##

The header only e3vector.h should be inluded in the main module.

No shared library, static library nor executable is installed,
because all functions in this class are provided as inline functions.


```cpp
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
```

## License ##

<a href="LICENSE">LGPLv2.1+</a>.


---
EOF





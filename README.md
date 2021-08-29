

# Latin Squares
[![Latest PyPI version](https://img.shields.io/pypi/v/latinsq?logo=pypi)](https://pypi.python.org/pypi/latinsq)
[![GitHub Workflow Status - Test](https://img.shields.io/github/workflow/status/ianabc/latinsq/Run%20Tests?logo=github&label=tests)](https://github.com/ianabc/latinsq/actions)
[![codecov](https://codecov.io/gh/ianabc/latinsq/branch/main/graph/badge.svg?token=SWUG5R4WR6)](https://codecov.io/gh/ianabc/latinsq)

This is a small package for generating and manipulating [Latin
Squares](https://en.wikipedia.org/wiki/Latin_square). It was created to support
games an other applications for the [Callysto Project](https://callysto.ca). It
hasn't been particularly widely tested or optimized so use at your own risk.

## Getting Started
This package implements a `LatinSquare` object with various methods to
initialize, generate new squares, check validity and things like that.

```python
>>> import numpy as np
>>> # from src.latinsq import LatinSquare
>>> from latinsq import LatinSquare


>>> # Initialize from values
>>> square = np.array([
[1, 2, 3],
[3, 1, 2],
[2, 3, 1]
])
>>> ls = LatinSquare(square)
array([[1, 2, 3],
       [3, 1, 2],
       [2, 3, 1]])

>>> square.valid()
True

>>> # Generate a random square
>>> LatinSquare.random(n=3)
array([[2, 1, 3],
       [3, 2, 1],
       [1, 3, 2]])
```

The random generator attempts to implement an algorithm described by [Jacobson
and
Matthews](https://doi.org/10.1002/(SICI)1520-6610(1996)4:6%3C405::AID-JCD3%3E3.0.CO;2-J)
draw a random latin square uniformly from the space of valid latin squares.

## References

1. Wikipedia article: [Latin Squares](https://en.wikipedia.org/wiki/Latin_square)
1. Generating Uniformly distributed random latin squares - _Jacobson & Matthews_
[doi:10.1002/(sici)1520-6610(1996)4:6<405::aid-jcd3>3.0.co;2-j](https://doi.org/10.1002%2F%28sici%291520-6610%281996%294%3A6%3C405%3A%3Aaid-jcd3%3E3.0.co%3B2-j)
1. [susansmathgames.ca](https://susansmathgames.ca) has an article on [latin and
euler squares](https://susansmathgames.ca/posts/latin-euler-squares/) with good
exposition on teaching with squares.
1. Knuth, The Art of Computer Programming Volume 4A supposedly has a
discussion. I haven't seen it yet, but will probably be interesting.
1. [Wolfram.com latin squares](https://mathworld.wolfram.com/LatinSquare.html).
1. [Enclopedia of Mathematics](https://encyclopediaofmath.org/wiki/Latin_square).
1. [Blog article with go implementation](https://blog.paulhankin.net/latinsquares/). This is probably closest to the version implemented here.
1. [SageMath Latin Square
Implementation](https://doc.sagemath.org/html/en/reference/combinat/sage/combinat/matrices/latin.html#sage.combinat.matrices.latin.LatinSquare_generator) - A recent find, but if we can split out the dependencies might be very useful.



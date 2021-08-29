# Latin Squares

This is a small package for generating and manipulating [Latin
Squares](https://en.wikipedia.org/wiki/Latin_square). It was created to support
games an other applications for the [Callysto Project](https://callysto.ca).

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

>>>square.valid()
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


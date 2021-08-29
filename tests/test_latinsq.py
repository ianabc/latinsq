import numpy as np
from numpy.testing import assert_equal
from ..src.latinsq import LatinSquare


def test_init_empty_square():
    l = LatinSquare(n=4)
    assert type(l.square) == np.ndarray
    assert l.square.shape == (4, 4)
    assert_equal(l.square, np.zeros((4, 4), np.int8))


def test_init_from_ndarray():
    s = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]], dtype=np.int8)
    l = LatinSquare(square=s)
    assert type(l.square) == type(s)
    print(l.square)
    print(s)
    assert_equal(l.square, s)


def test_init_from_random():
    r = LatinSquare.random(n=3, rng_seed=12345)
    assert_equal(np.array([[2, 3, 1], [3, 1, 2], [1, 2, 3]]), r.square)


def test_latinsq_repr():
    l = LatinSquare(square=np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]], dtype=np.int8))
    assert (
        repr(l)
        == "array([[1, 2, 3],\n       [2, 3, 1],\n       [3, 1, 2]], dtype=int8)"
    )


def test_latinsq_getitem():
    l = LatinSquare(square=np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]], dtype=np.int8))
    assert l[0, 1] == 2


def test_latinsq_setitem():
    l = LatinSquare(n=3)
    l[1, 2] = 3
    assert l[1, 2] == 3


def test_valid_is_valid():
    l = LatinSquare(square=np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]], dtype=np.int8))
    assert l.valid()


def test_invalid_is_not_valid():
    l = LatinSquare(square=np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]], dtype=np.int8))
    l[0, 0] = 3
    assert not l.valid()


def test_init_from_random_no_seed():
    r = LatinSquare.random(n=4)
    assert type(r.square) == np.ndarray
    assert r.valid()


def test_empty_is_not_valid():
    l = LatinSquare(n=4)
    assert not l.valid()


def test_as_rcv():
    l = LatinSquare(square=np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]], dtype=np.int8))
    r, c, v = l.as_rcv()
    assert type(r) == type(c) == type(v) == np.ndarray
    assert len(r) == len(c) == len(v) == l.n * l.n
    assert_equal(np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int8), r)
    assert_equal(np.array([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.int8), c)
    assert_equal(np.array([1, 2, 3, 2, 3, 1, 3, 1, 2], dtype=np.int8), v)


def test_from_rcv():
    r = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int8)
    c = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.int8)
    v = np.array([1, 2, 3, 2, 3, 1, 3, 1, 2], dtype=np.int8)
    s = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
    l1 = LatinSquare(square=s)
    l2 = LatinSquare.from_rcv(r, c, v)
    # We should really define __equality__ for LatinSquares
    # And test different orders of r, c and v
    assert_equal(l1.square, l2.square)


def test_to_incidence_matrix():
    square = np.array([[2, 1, 3], [1, 3, 2], [3, 2, 1]], dtype=np.int8)
    inc_matrix = LatinSquare.to_incidence_matrix(square)
    t_inc_matrix = np.array(
        [
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        ],
        dtype=np.int8,
    )
    assert_equal(t_inc_matrix, inc_matrix)


def test_as_incidence_matrix():
    l = LatinSquare(square=np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]], dtype=np.int8))
    assert_equal(l.as_incidence_matrix(), LatinSquare.to_incidence_matrix(l.square))


def test_from_incidence_matrix():
    l = LatinSquare(square=np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]], dtype=np.int8))
    inc_matrix = np.array(
        [
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        ],
        dtype=np.int8,
    )
    assert_equal(LatinSquare.from_incidence_matrix(inc_matrix).square, l.square)


def test_is_incidence_matrix():
    inc_matrix = np.array(
        [
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        ],
        dtype=np.int8,
    )
    assert LatinSquare.is_incidence_matrix(inc_matrix)


def test_valid_is_valid_incidence_matrix():
    inc_matrix = np.array(
        [
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        ],
        dtype=np.int8,
    )
    assert LatinSquare.is_incidence_matrix(inc_matrix)


def test_invalid_is_not_valid_incidence_matrix():
    inc_matrix = np.array(
        [
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        ],
        dtype=np.int8,
    )
    inc_matrix[0, 0, 0] = 0
    assert not LatinSquare.is_incidence_matrix(inc_matrix)

import numpy as np


class LatinSquare(object):
    """
    A latin square object

    Internally, we represent an n x n latin square as a 3 x n**2 array,
    specifying the row index, column index and value for each position. This
    makes some of the solver manipulations significanlty easier. For example

      1 3 2
      2 1 3
      3 2 1

    Is stored as

      R: 111222333
      C: 123123123
      V: 132213321

    Which should be read as the value in rown 1, position 2 has value 3 etc.
    (we will index from 1 throughout)
    """
    def __init__(self, n=3, square=None):
        if square:
            assert type(square) == np.ndarray
            assert len(square.shape) == 2
            assert square.shape[0] == square.shape[1]
            self.n = square.shape[0]
            self.square = square
        else:
            self.n = n
            self.random()

    def __repr__(self):
        return repr(self.square)

    def __getitem__(self, idx):
        return self.square[idx[0], idx[1]]

    def __setitem__(self, idx, value):
        self.square[idx[0], idx[1]] = value

    def as_rcv(self):
        r = np.repeat(np.arange(1, self.n + 1), self.n)
        c = np.arange(self.n * self.n) % self.n + 1
        v = np.ravel(self.square)

        return (r, c, v)

    def valid(self):
        s = np.arange(3, dtype=np.int8)
        s = s * np.ones(self.n, dtype=np.int8)[:, np.newaxis] + 1

        return (
            np.all(np.sort(self.square, axis=0) == s.T) and
            np.all(np.sort(self.square, axis=1) == s)
        )

    @staticmethod
    def from_rcv(r, c, v):
        r = np.array(r)
        c = np.array(c)
        v = np.array(v)
        n = int(np.sqrt(len(v)))
        square = LatinSquare(n=n)
        assert len(r) == len(c) == len(v)
        assert len(r) == int(np.sqrt(len(r)))**2
        square.square = np.ones((n, n), dtype=np.int8) * -1
        square.square[
            r.reshape(n, n) - 1,
            c.reshape(n, n) - 1
        ] = v.reshape(n, n)
        return square

    def as_incidence_matrix(self):
        # Generate the equivalent 3D incidence matrix
        xy = np.mgrid[0:self.n, 0:self.n]
        idx = np.array([xy[0], xy[1], self.square - 1])
        inc_matrix = np.zeros((self.n, self.n, self.n), dtype=np.int8)
        inc_matrix[idx[2], idx[0], idx[1]] = 1
        return inc_matrix

    @staticmethod
    def from_incidence_matrix(inc):
        assert type(inc) == np.ndarray
        assert inc.ndim == 3
        n = inc.shape[0]
        square = LatinSquare(n=n)
        s = np.arange(n)[:, np.newaxis, np.newaxis] + 1
        s = (s * inc).cumsum(axis=0)[-1]
        square.square = s
        return square

    def random(self):
        """
        Generate a random latin square

        Use MCMC to sample from the space of valid latin squares
        """
        self.square = np.ones((self.n, self.n), dtype=np.int8) * -1
        for idx, row in enumerate(self.square):
            self.square[idx, :] = np.roll(np.arange(1, self.n + 1), idx)

        # Just a shuffle, not really random, result is in same
        rng = np.random.default_rng()
        self.square = self.square[rng.permutation(self.n)]
        self.square = self.square[:, rng.permutation(self.n)]
        self._i_matrix = self.as_incidence_matrix()


if __name__ == "__main__":
    sq = LatinSquare()
    print("Latin Square: ", sq)

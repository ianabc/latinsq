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
    def __init__(self, square=None, n=3):
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

    def to_rcv(self):
        r = np.repeat(np.arange(1, self.n + 1), self.n)
        c = np.arange(self.n * self.n) % 3 + 1
        v = np.ravel(self.square)

        return (r, c, v)

    def valid(self):
        s = np.arange(3, dtype=np.int8)
        s = s * np.ones(self.n, dtype=np.int8)[:, np.newaxis] + 1

        return (
            np.all(np.sort(self.square, axis=0) == s.T) and
            np.all(np.sort(self.square, axis=1) == s)
        )

    def from_rcv(self, r, c, v):
        r = np.array(r)
        c = np.array(c)
        v = np.array(v)
        self.square = np.ones((3, 3), dtype=np.int8) * -1
        self.square[r.reshape(3, 3) - 1, c.reshape(3, 3) - 1] = v.reshape(3, 3)

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

        # Generate the equivalent 3D incidence matrix
        xy = np.mgrid[0:self.n, 0:self.n]
        idx = np.array([xy[0], xy[1], self.square - 1])
        self._i_matrix = np.zeros((self.n, self.n, self.n), dtype=np.int8)
        self._i_matrix[idx[2], idx[0], idx[1]] = 1


if __name__ == "__main__":
    sq = LatinSquare()
    print("Latin Square: ", sq)

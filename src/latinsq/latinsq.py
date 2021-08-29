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

    def __init__(self, n=3, square=None, generate=False):
        if square is not None:
            assert type(square) == np.ndarray
            assert len(square.shape) == 2
            assert square.shape[0] == square.shape[1]
            self.n = square.shape[0]
            self.square = square
        elif generate:
            self.n = n
            self.square = LatinSquare.random(self.n)
        else:
            self.n = n
            self.square = np.zeros((self.n, self.n), dtype=np.int8)

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
        s = np.arange(self.n, dtype=np.int8)
        s = s * np.ones(self.n, dtype=np.int8)[:, np.newaxis] + 1

        return np.all(np.sort(self.square, axis=0) == s.T) and np.all(
            np.sort(self.square, axis=1) == s
        )

    @staticmethod
    def from_rcv(r, c, v):
        r = np.array(r)
        c = np.array(c)
        v = np.array(v)
        n = int(np.sqrt(len(v)))
        square = LatinSquare(n=n)
        assert len(r) == len(c) == len(v)
        assert len(r) == int(np.sqrt(len(r))) ** 2
        square.square = np.ones((n, n), dtype=np.int8) * -1
        square.square[r.reshape(n, n) - 1, c.reshape(n, n) - 1] = v.reshape(n, n)
        return square

    def to_incidence_matrix(square):
        # Generate the equivalent 3D incidence matrix
        assert type(square) == np.ndarray
        assert square.ndim == 2
        assert square.shape[0] == square.shape[1]
        n = square.shape[0]
        xy = np.mgrid[0:n, 0:n]
        idx = np.array([xy[0], xy[1], square - 1])
        inc_matrix = np.zeros((n, n, n), dtype=np.int8)
        inc_matrix[idx[2], idx[0], idx[1]] = 1
        return inc_matrix

    def as_incidence_matrix(self):
        # Generate the equivalent 3D incidence matrix
        return LatinSquare.to_incidence_matrix(self.square)

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

    @staticmethod
    def is_incidence_matrix(v):
        """Includes valid and almost incidence matrices"""
        return (
            (len(v.shape) == 3)
            and (v.shape[0] == v.shape[1] == v.shape[2])
            and np.all((v.sum(axis=0) == np.ones_like(v)))
            and np.all((v.sum(axis=1) == np.ones_like(v)))
            and np.all((v.sum(axis=2) == np.ones_like(v)))
        )

    @staticmethod
    def is_valid_incidence_matrix(v):
        """Must be an incidence matrix AND be valid"""
        u = np.unique(v)
        return LatinSquare.is_incidence_matrix(v) and (len(u) == 2) and (min(u) == 0)

    @staticmethod
    def random(n):
        """
        Generate a random latin square

        Use MCMC to sample from the space of valid latin squares
        """
        square = np.ones((n, n), dtype=np.int8) * -1
        for idx, row in enumerate(square):
            square[idx, :] = np.roll(np.arange(1, n + 1), idx)

        # Just a shuffle, not really random, result is in same
        rng = np.random.default_rng()
        square = square[rng.permutation(n)]
        square = square[:, rng.permutation(n)]

        # Jacobson and Matthews MCMC algorithm operates on
        # the equivalent incidence matrix
        inc_matrix = LatinSquare.to_incidence_matrix(square)

        # 1. If M is valid incidence marix, choose some i, j, k with M_ijk = 0;
        # If M is an almost incidence matrix, choose i, j, k such that M_ijk=-1
        # 2. Let r, s, t be indices such that M_rjk = M_isk = M_ijt = 1. If M
        # is valid, this choise is unique, if it is "almost" there are two
        # possible choices for each index.
        # 3. Increase M_ijk, M_irt, M_rjt and M_rsk by 1, and decrease M_rjk,
        # M_irk, M_ijt and M_rst by one
        #
        # The result will be another actual or almost incidence matrix
        # depending on whether M_rst is 1 or 0.
        #
        # The transition step is to add 1 to M_ijk, subtract one from M_rst and
        # toggle the remaining entries. If M_rst was 1, this will result in a
        # valid incidence matrix. If not, it will give an almost matrix and we
        # go again.
        #
        # In either case the idea is to take enough steps to ensure we're
        # sampling uniformly and to only yield from the iterator on a valid
        # matrix.
        iterations = 0
        while (iterations < n * n * n) or not LatinSquare.is_valid_incidence_matrix(
            inc_matrix
        ):
            iterations = iterations + 1
            if LatinSquare.is_valid_incidence_matrix(inc_matrix):
                while True:
                    i, j, k = rng.integers(n, size=3)
                    if inc_matrix[i, j, k] == 0:
                        break

                i1 = np.argmax(inc_matrix[:, j, k])
                j1 = np.argmax(inc_matrix[i, :, k])
                k1 = np.argmax(inc_matrix[i, j, :])

            else:
                # Indices of -1 entry
                i, j, k = np.unravel_index(
                    np.argmin(inc_matrix, axis=None), inc_matrix.shape
                )

                # Each line passing through (i, j, k) has two 1 values. Find
                # both and randomly select one.
                i1 = rng.choice(np.where(inc_matrix[:, j, k] == 1)[0])
                j1 = rng.choice(np.where(inc_matrix[i, :, k] == 1)[0])
                k1 = rng.choice(np.where(inc_matrix[i, j, :] == 1)[0])

            inc_matrix[i, j, k] += 1
            inc_matrix[i1, j1, k] += 1
            inc_matrix[i1, j, k1] += 1
            inc_matrix[i, j1, k1] += 1

            inc_matrix[i1, j1, k1] -= 1
            inc_matrix[i1, j, k] -= 1
            inc_matrix[i, j1, k] -= 1
            inc_matrix[i, j, k1] -= 1

        return LatinSquare.from_incidence_matrix(inc_matrix)


if __name__ == "__main__":
    sq = LatinSquare()
    print("Latin Square: ", sq)

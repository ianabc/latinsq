import numpy as np

class LatinSquare(object):
    """
    A latin square object

    Internally, we represent an n x n latin square as a 3 x n**2 array, specifying
    the row index, column index and value for each position. This makes some of the
    solver manipulations significanlty easier. For example

      1 3 2
      2 1 3
      3 2 1

    Is stored as

      R: 111222333
      C: 123123123
      V: 132213321
    
    Which should be read as the value in rown 1, position 2 has value 3 etc. (we will 
    index from 1 throughout)
    """
    def __init__(self, square = None, n=3):
        if square:
            assert type(square) == np.ndarray
            assert len(square.shape) == 2
            assert square.shape[0] == share.shape[1]
            self.n = square.shape[0]
            self.square = square
        else:
            self.n = n
            self.square = self.random(self.n)

    
    def __repr__(self):
        return self.square.__repr__()


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
        self.square[r.reshape(3, 3) - 1, c.reshape(3, 3) - 1] = v.reshape(3,3)


    def random(self, n):
        """
        Generate a random latin square

        Use MCMC to sample from the space of valid latin squares
        """
        square = np.ones((n, n), dtype=np.int8) * -1
        return square


if __name__ == "__main__":
    l = LatinSquare()
    print("Latin Square: ", l)
import numpy as np

from rllab.misc import ext, special
from rllab.spaces.base import Space


class DiscreteSequence(Space):
    """
    {0,1,...,n-1}^k
    """

    def __init__(self, n, k):
        self._n = n
        self._k = k

    @property
    def n(self):
        return self._n

    def sample(self):
        return np.random.randint(self._n, size=self._k)

    def contains(self, x):
        x = np.asarray(x)
        return (x.shape == (self._n,) and x.dtype.kind == 'i'
                and (x >= 0 and x < self._n).all())

    def __repr__(self):
        return "DiscreteSequence(%i, %i)" % (self._n, self._k)

    def __eq__(self, other):
        return self._n == other._n and self._k == other._k

    def __hash__(self):
        return hash(self._n) ^ hash(self._k)

    def flatten(self, x):
        # HACK: always work on flattened representation
        if x.ndim < 2:
            return x
        return special.from_onehot_n(x)

    def unflatten(self, x):
        return x# special.to_onehot_n(x, self.n)

    def flatten_n(self, xs):
        print xs
        raise NotImplementedError

    def unflatten_n(self, xs):
        print xs
        raise NotImplementedError

    @property
    def flat_dim(self):
        return self._n * self._k

    def weighted_sample(self, weights):
        # TODO not sure if this is used?
        raise NotImplementedError

    def new_tensor_variable(self, name, extra_dims):
        """
        Return a batch representing values from this space.
        `extra_dims` may be used to add e.g. recurrent / batch axes.
        """
        return ext.new_tensor(
                name=name,
                ndim=extra_dims + 1,
                dtype="uint8"
            )

from nose.tools import eq_

import numpy as np
import tensorflow as tf
# Don't care about tf in these tests
tf.logging.set_verbosity(tf.logging.ERROR)

from praglang.agents.grid import GridWorldMasterAgent
from praglang.environments.grid import SlaveGridWorldEnv


def test_where():
    maps = [(["GFFFFFFFFFFFFS"], "w"),
            (["SFFFFFFFFFFFFG"], "e"),
            (["S", "F", "F", "G"], "s"),
            (["G", "F", "F", "S"], "n"),]

    for map, direction in maps:
        e = SlaveGridWorldEnv(map)
        e.reset()
        a = GridWorldMasterAgent(e)

        query = "h"
        query = [a.token2idx[t] for t in query]
        response, _ = a(e, query)
        response = "".join([a.vocab[idx] for idx in response])

        yield eq_, response, direction


def test_wall():
    maps = [(["WWWWW",
              "WWWWW",
              "GWSWF",
              "WWWWW",
              "WWWWW"],
             "ww",
             ["WWWWW",
              "WWWWW",
              "GFSWF",
              "WWWWW",
              "WWWWW"]),]

    for map_start, query, result in maps:
        e = SlaveGridWorldEnv(map_start)
        e.reset()
        a = GridWorldMasterAgent(e)

        query = [a.token2idx[t] for t in query]
        response, _ = a(e, query)

        np.testing.assert_array_equal(e.map_desc, np.array(map(list, result)).T)


import unittest
from textwrap import dedent

from qspectra import utils


class Example(object):
    def __init__(self, a=0, b=1, c=None, d=None):
        self.a = a
        self.b = b
        self.c = self if c is None else c
        self.d = Example(1, 2, 3, 4) if d is None else d

    def __repr__(self):
        return utils.inspect_repr(self)


class TestInspectRepr(unittest.TestCase):
    def test(self):
        print(Example())
        self.assertEqual(
            repr(Example()),
            dedent("""
            Example(
                a=0,
                b=1,
                c=Example(...),
                d=Example(
                      a=1,
                      b=2,
                      c=3,
                      d=4))
            """).strip())

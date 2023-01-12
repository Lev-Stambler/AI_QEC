from CPC.cpc_code import *
from CPC import generate_random
import unittest


class TestCPCSimplifications(unittest.TestCase):
    def test_generate_random(self):
        code = generate_random.random_cpc(80, 50, 5, 5, 3)
        print("Got code", code)
        # code.simplify()
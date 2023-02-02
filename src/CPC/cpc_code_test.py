# TODO: UPDATE ME TO MOST RECENT INSTANTIATION
from CPC.cpc_code import *
import unittest


class TestCPCSimplifications(unittest.TestCase):
    def test_rule_1(self):
        data_vert = CPCVertex(0, data_qubit=True)
        check_1 = CPCVertex(1, check_qubit=True)
        check_2 = CPCVertex(2, check_qubit=True)
        c = CPCCode([
            CPCEdge(data_vert, check_1, bit_check=True),
            CPCEdge(data_vert, check_2, bit_check=False)
        ])
        c.simplify_virtual_edges()
        self.assertEqual(len(c.edges), 3)
        self.assertListEqual([
            CPCEdge(data_vert, check_1, bit_check=True),
            CPCEdge(data_vert, check_2, bit_check=False),
            CPCEdge(check_1, check_2, virtual_edge=True),
        ], c.edges)

		# Mirrored now
        data_vert = CPCVertex(0, data_qubit=True)
        check_1 = CPCVertex(1, check_qubit=True)
        check_2 = CPCVertex(2, check_qubit=True)
        c = CPCCode([
            CPCEdge(data_vert, check_1, bit_check=False),
            CPCEdge(data_vert, check_2, bit_check=True),
        ])
        c.simplify_virtual_edges()
        self.assertEqual(len(c.edges), 3)
        self.assertListEqual([
            CPCEdge(data_vert, check_1, bit_check=False),
            CPCEdge(data_vert, check_2, bit_check=True),
            CPCEdge(check_2, check_1, virtual_edge=True),
        ], c.edges)

    def test_rule_2(self):
        data_vert = CPCVertex(0, data_qubit=True)
        check_1 = CPCVertex(1, check_qubit=True)
        c = CPCCode([
            CPCEdge(check_1, data_vert, bit_check=True),
            CPCEdge(check_1, data_vert, bit_check=False),
        ])
        c.simplify_virtual_edges()

        self.assertListEqual([
            CPCEdge(check_1, data_vert, bit_check=True),
            CPCEdge(check_1, data_vert, bit_check=False),
            CPCEdge(check_1, check_1, virtual_edge=True),
        ], c.edges)
        
        # Test mirrored version
        c = CPCCode([
            CPCEdge(data_vert, check_1, bit_check=True),
            CPCEdge(data_vert, check_1, bit_check=False),
        ])
        c.simplify_virtual_edges()

        self.assertListEqual([
            CPCEdge(data_vert, check_1, bit_check=True),
            CPCEdge(data_vert, check_1, bit_check=False),
            CPCEdge(check_1, check_1, virtual_edge=True),
        ], c.edges)
        
        

    def test_rule_3(self):
        check_1 = CPCVertex(0, check_qubit=True)
        check_2 = CPCVertex(1, check_qubit=True)
        c = CPCCode([
            CPCEdge(check_2, check_1, virtual_edge=True),
            CPCEdge(check_2, check_1, virtual_edge=True)
        ])
        c.simplify_virtual_edges()

        self.assertCountEqual([], c.edges)

    def test_rule_4(self):
        check_1 = CPCVertex(0, check_qubit=True)
        c = CPCCode([
            CPCEdge(check_1, check_1, virtual_edge=True),
            CPCEdge(check_1, check_1, virtual_edge=True)
        ])
        c.simplify_virtual_edges()

        self.assertCountEqual([], c.edges)

    def test_rule_5(self):
        check_1 = CPCVertex(0, check_qubit=True)
        check_2 = CPCVertex(1, check_qubit=True)
        c = CPCCode([
            CPCEdge(check_2, check_1, virtual_edge=True),
            CPCEdge(check_2, check_1, bit_check=True),
        ])
        c.simplify_virtual_edges()
        self.assertListEqual([CPCEdge(check_1, check_2, virtual_edge=True)], c.edges)

    def test_rule_6(self):
        check_1 = CPCVertex(0, check_qubit=True)
        check_2 = CPCVertex(1, check_qubit=True)
        c = CPCCode([
            CPCEdge(check_1, check_2, virtual_edge=True),
            CPCEdge(check_1, check_2, bit_check=True),
        ])
        c.simplify_virtual_edges()
        self.assertListEqual([CPCEdge(check_1, check_2, bit_check=True)], c.edges)


if __name__ == '__main__':
    unittest.main()

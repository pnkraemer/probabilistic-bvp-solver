from bvps.quadrature import *


def test_gauss_lobatto_interior_only():
    rule = gauss_lobatto_interior_only()
    assert rule.nodes.shape == (3,)
    assert rule.weights.shape == (3,)

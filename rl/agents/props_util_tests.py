import unittest
import copy
from props_util import *
import numpy as np
import numpy.testing as np_testing
import timeit

class TestKL(unittest.TestCase):
    def test_same_dist(self):
        pk0 = NormalDist(np.zeros(3), np.identity(3))
        pk1 = NormalDist(np.zeros(3), np.identity(3))
        self.assertEqual(kl(pk0, pk1), 0)

    def test_diff_mean(self):
        pk0 = NormalDist(np.zeros(3), np.identity(3))
        pk1 = NormalDist(np.ones(3), np.identity(3))
        self.assertEqual(kl(pk0, pk1), 3/2.0)

    def test_diff_mean_cov(self):
        pk0 = NormalDist(np.zeros(3), np.identity(3))
        pk1 = NormalDist(np.ones(3), 2*np.identity(3))
        self.assertAlmostEqual(kl(pk0, pk1), 2.07944154/2.0)

class TestKLGrad(unittest.TestCase):
    def finite_diff(self, p0, p1):
        eps = 1e-5
        drdm_fd = np.zeros(3)
        drdS_fd = np.zeros(3)
        for i in range(0, 3):
            p0_m_fd1 = copy.deepcopy(p0)
            p0_m_fd1.m[i] = p0_m_fd1.m[i] + eps
            p0_m_fd2 = copy.deepcopy(p0)
            p0_m_fd2.m[i] = p0_m_fd2.m[i] - eps
            drdm_fd[i] = (kl(p0_m_fd1, p1) - kl(p0_m_fd2, p1))/(2*eps)

            p0_S_fd1 = copy.deepcopy(p0)
            p0_S_fd1.S[i,i] = p0_S_fd1.S[i,i] + eps
            p0_S_fd2 = copy.deepcopy(p0)
            p0_S_fd2.S[i,i] = p0_S_fd2.S[i,i] - eps
            drdS_fd[i] = (kl(p0_S_fd1, p1) - kl(p0_S_fd2, p1))/(2*eps)
        return drdm_fd, drdS_fd
   
    def compare_grad(self, p0, p1):
        drdm, drdS = kl_grad(p0, p1)
        drdm_fd, drdS_fd = self.finite_diff(p0, p1)
        np_testing.assert_allclose(drdm, drdm_fd)
        np_testing.assert_allclose(drdS, drdS_fd, atol=1e-8)

    def test_same_dist(self):
        pk0 = NormalDist(np.zeros(3), np.identity(3))
        pk1 = NormalDist(np.zeros(3), np.identity(3))
        self.compare_grad(pk0, pk1)

    def test_diff_mean(self):
        pk0 = NormalDist(np.zeros(3), np.identity(3))
        pk1 = NormalDist(np.ones(3), np.identity(3))
        self.compare_grad(pk0, pk1)

    def test_diff_mean_cov(self):
        pk0 = NormalDist(np.zeros(3), np.identity(3))
        pk1 = NormalDist(np.ones(3), 2*np.identity(3))
        self.compare_grad(pk0, pk1)

    def test_diff_mean_cov2(self):
        pk0 = NormalDist(1.34*np.ones(3), 1.522*np.identity(3))
        pk1 = NormalDist(np.ones(3), 2*np.identity(3))
        self.compare_grad(pk0, pk1)

    def test_rand_mean_cov(self):
        np.random.seed(0)
        pk0 = NormalDist(np.random.rand(3), np.identity(3))
        pk1 = NormalDist(np.random.rand(3), 2*np.identity(3))
        self.compare_grad(pk0, pk1)

if __name__ == '__main__':
    unittest.main()

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

class TestBoundAll(unittest.TestCase):
    def test_rand_input(self):
        Jss = np.ones((4,10))
        wss = 2*np.ones((4,10))
        rdas = np.array([[0.2], [0.4], [0.6], [0.8]])

        Jrob, Jha, _ = bound_all_2(wss, Jss, rdas)
        self.assertAlmostEqual(Jha, 2)
        self.assertAlmostEqual(Jrob, 4)

class TestWithPolicyData(unittest.TestCase):
    def setUp(self):
        self.d = 3
        self.L = 4
        self.M = 10
        self.pk = NormalDist(np.array([0.4509, 0.5470, 0.2963]), 1.0745*np.identity(self.d))
        
        self.pkms = np.array([[[0.4387],[0.3816],[0.7655]],
               [[0.3500],[0.1966],[0.2511]],
               [[0.7482],[0.4505],[0.0838]],
               [[0.0760],[0.2399],[0.1233]]])
        self.pks = []
        for i in range(0, self.L):
            self.pks.append(NormalDist(np.zeros(self.d), np.identity(self.d)))
            self.pks[i].m = self.pkms[i,:,:].flatten()
            
        self.pks[0].S = 1.0795*np.identity(self.d)
        self.pks[1].S = 1.0616*np.identity(self.d)
        self.pks[2].S = 1.0229*np.identity(self.d)
        self.pks[3].S = 1.0184*np.identity(self.d)

        self.Js = np.array(
          [[0.8147, 0.6324, 0.9575, 0.9572, 0.4218, 0.6557, 0.6787, 0.6555, 0.2769, 0.6948],
           [0.9058, 0.0975, 0.9649, 0.4854, 0.9157, 0.0357, 0.7577, 0.1712, 0.0462, 0.3171],
           [0.1270, 0.2785, 0.1576, 0.8003, 0.7922, 0.8491, 0.7431, 0.7060, 0.0971, 0.9502],
           [0.9134, 0.5469, 0.9706, 0.1419, 0.9595, 0.9340, 0.3922, 0.0318, 0.8235, 0.0344]])

        self.rdas = np.zeros(self.L)
        self.Jmaxs = np.zeros(self.L)
        for i in range(0, self.L):
            self.rdas[i] = renyii(self.pk, self.pks[i], 2)
            self.Jmaxs[i] = max(self.Js[i,:])

        self.ks = np.array([[[-0.4599, 0.1672, 2.1182, -1.0910],[1.0909, 0.6504, 0.8736, 1.0490],[-0.4586, -0.4789, 0.5482, 0.0409],[-0.7182, 0.0989, 1.5828, -0.0585],[-0.3609, -0.8328, 0.2742, -0.1509],[1.5997, 3.0139, 0.4671, -0.7797],[1.0128, 0.2650, 0.3903, 1.7466],[0.5280, -1.4991, 1.2619, -0.7974],[-0.6642, 0.4531, -0.6007, -1.2682],[1.2160, -0.2686, 0.4457, 0.4125]],[[0.3503, -0.0054, -0.6338, -0.2985],[1.5175, 0.4004, 1.9036, 0.7647],[0.4619, 0.9144, -0.7711, -0.5656],[0.3744, 0.4188, 1.8452, -0.4812],[0.7674, 0.3047, 0.1750, -0.3545],[-0.7500, -0.4905, 1.1601, -0.8905],[1.5251, -1.7951, -0.3824, 0.5503],[-1.1682, 1.0625, 0.7357, 0.0618],[2.8237, -0.3645, 1.5909, -2.1113],[0.1816, 0.7014, 0.4737, 0.6349]],[[0.5942, 1.7135, 1.0557, -1.8976],[1.9181, 1.8870, -1.8994, 0.1031],[-0.4960, 1.1115, 3.0249, 1.1513],[2.3579, -0.9501, -0.9864, 1.4871],[0.5311, 0.9953, 1.1948, -0.1731],[0.7993, 0.4441, -1.9914, 2.6724],[2.3700, -0.2012, -1.5112, -1.1453],[-0.0057, -0.6639, 0.1177, 0.9220],[0.1259, 0.5638, 0.4380, -1.3390],[1.6888, 1.0129, -0.1812, 0.5791]]])

        
class TestDistJhaGrad(TestWithPolicyData):
    def finite_diff(self, pk):
        eps = 1e-5
        djdm_fd = np.zeros(3)
        djdS_fd = np.zeros(3)
        djda_fd = 0
        for i in range(0, 3):
            pk_m_fd1 = copy.deepcopy(pk)
            pk_m_fd1.m[i] = pk_m_fd1.m[i] + eps
            pk_m_fd2 = copy.deepcopy(pk)
            pk_m_fd2.m[i] = pk_m_fd2.m[i] - eps
            djdm_fd[i] = (dist_jha_2(pk_m_fd1, self.pks, self.Js, self.ks) - 
                dist_jha_2(pk_m_fd2, self.pks, self.Js, self.ks))/(2*eps)

            pk_S_fd1 = copy.deepcopy(pk)
            pk_S_fd1.S[i,i] = pk_S_fd1.S[i,i] + eps
            pk_S_fd2 = copy.deepcopy(pk)
            pk_S_fd2.S[i,i] = pk_S_fd2.S[i,i] - eps
            djdS_fd[i] = (dist_jha_2(pk_S_fd1, self.pks, self.Js, self.ks) - 
                dist_jha_2(pk_S_fd2, self.pks, self.Js, self.ks))/(2*eps)
        return djdm_fd, djdS_fd

    def compare_grad(self, pk):
        djdm, djdS = dist_jha_grad_2(pk, self.pks, self.Js, self.ks,
            {'normalize_weights' : True})
        djdm_fd, djdS_fd = self.finite_diff(pk)
        np_testing.assert_allclose(djdm, djdm_fd)
        np_testing.assert_allclose(djdS, djdS_fd, atol=1e-8)

    def test_rand_input(self):
        self.compare_grad(self.pk)
        self.compare_grad(self.pk)

class TestDistBoundRobustGrad(TestWithPolicyData):
    def finite_diff(self, opts):
        eps = 1e-5
        djdm_fd = np.zeros(3)
        djdS_fd = np.zeros(3)
        for i in range(0, 3):
            pk_m_fd1 = copy.deepcopy(self.pk)
            pk_m_fd1.m[i] = pk_m_fd1.m[i] + eps
            pk_m_fd2 = copy.deepcopy(self.pk)
            pk_m_fd2.m[i] = pk_m_fd2.m[i] - eps
            djdm_fd[i] = (
                dist_bound_2(pk_m_fd1, self.pks, self.Js, self.ks, self.L, opts)[0] - 
                dist_bound_2(pk_m_fd2, self.pks, self.Js, self.ks, self.L, opts)[0]) / (2*eps)

            pk_S_fd1 = copy.deepcopy(self.pk)
            pk_S_fd1.S[i,i] = pk_S_fd1.S[i,i] + eps
            pk_S_fd2 = copy.deepcopy(self.pk)
            pk_S_fd2.S[i,i] = pk_S_fd2.S[i,i] - eps
            djdS_fd[i] = (
                dist_bound_2(pk_S_fd1, self.pks, self.Js, self.ks, self.L, opts)[0] - 
                dist_bound_2(pk_S_fd2, self.pks, self.Js, self.ks, self.L, opts)[0]) / (2*eps)
        return djdm_fd, djdS_fd

    def compare_grad(self, opts):
        djdm, djdS = dist_bound_grad_2(
            self.pk, self.pks, self.Js, self.ks, self.rdas, opts)
        djdm_fd, djdS_fd = self.finite_diff(opts)
        np_testing.assert_allclose(djdm, djdm_fd)
        np_testing.assert_allclose(djdS, djdS_fd, atol=1e-8)
        
    def test_rand_input_normalized(self):
        opts = {'normalize_weights': True}
        self.compare_grad(opts)
        self.compare_grad(opts)

    def test_rand_input_unnormalized(self):
        opts = {'normalize_weights': False}
        self.compare_grad(opts)
        self.compare_grad(opts)

    def test_rand_input_unnormalized_truncated(self):
        opts = {'normalize_weights' : False, 
            'truncate_weights' : True,
             'truncate_tresh' : 1.6}
        self.compare_grad(opts)
        self.compare_grad(opts)

    def test_rand_input_normalized_truncated(self):
        opts = {'normalize_weights' : True, 
            'truncate_weights' : True,
                'truncate_tresh' : 2.0}
        self.compare_grad(opts)
        self.compare_grad(opts)
        
if __name__ == '__main__':
    unittest.main()

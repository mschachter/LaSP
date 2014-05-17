import copy
import unittest

import numpy as np

import matplotlib.pyplot as plt
from scipy.interpolate import splev
from tools.memd import create_mirrored_spline
from tools.signal import find_extrema


class MEMDTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_mirror_spline(self):

        freqs = [2.0, 25.0, 40.0]
        sr = 1e3
        dt = 1.0 / sr
        t = np.arange(0.0, 1.0+dt, dt)
        s = np.zeros([len(t)])
        #add a few sine waves up
        for f in freqs:
            s += np.sin(2*np.pi*f*t)

        #identity maxima and minima
        mini,maxi = find_extrema(s)
        mini_orig = copy.copy(mini)
        maxi_orig = copy.copy(maxi)

        #extrapolate and build splines using mirroring
        low_spline, high_spline = create_mirrored_spline(mini, maxi, s)

        #evaluate splines
        ti = np.arange(len(t))
        low_env = splev(ti, low_spline)
        high_env = splev(ti, high_spline)
        env = (high_env + low_env) / 2.0

        plt.figure()
        plt.plot(t, s, 'k-', linewidth=2.0)
        plt.plot(t, low_env, 'b-')
        plt.plot(t, high_env, 'r-')
        plt.plot(t, env, 'g-')
        plt.plot(t[mini_orig], s[mini_orig], 'bo')
        plt.plot(t[maxi_orig], s[maxi_orig], 'ro')

        #plot extrapolated endpoints
        Tl = maxi[0] / sr
        tl = mini[0] / sr
        Tr = maxi[-1] / sr
        tr = mini[-1] / sr

        plt.plot(Tl, splev(Tl, high_spline), 'mo')
        plt.plot(tl, splev(tl, low_spline), 'co')
        plt.plot(Tr, splev(Tr, high_spline), 'mo')
        plt.plot(tr, splev(tr, low_spline), 'co')

        plt.axis('tight')

        plt.show()

    def test_mean_envelope(self):
        pass


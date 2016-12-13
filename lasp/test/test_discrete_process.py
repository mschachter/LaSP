from unittest import TestCase

import numpy as np
import matplotlib.pyplot as plt


class DiscreteProcessTest(TestCase):


    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_static_data(self):

        # specify the prior density and distribution
        ndim = 3
        p_prior = np.array([0.2, 0.7, 0.1])
        pdist = np.cumsum(p_prior)

        sample_rate = 200.
        duration = 60.
        dt = 1. / sample_rate
        nt = int(duration*sample_rate)

        # random event rate - events roughly every 150ms
        event_tau = 150e-3
        num_events = int(duration / event_tau)

        # generate events at each event time, drawing from the probability distribution, otherwise leave
        # the value equal to zero. zero means "no event", while an integer from 1-3 indicates an event
        # occurred
        s = np.zeros([nt], dtype='int')
        next_event = np.random.exponential(event_tau)
        t = np.arange(nt) / sample_rate
        for k,ti in enumerate(t):
            if ti >= next_event:
                # generate a sample from the distribution
                r = np.random.rand()
                i = np.min(np.where(pdist > r)[0])
                s[k] = i + 1
                next_event = ti + np.random.exponential(event_tau)

        num_events = np.sum(s > 0)
        print '# of events: %d (%0.3f Hz)' % (num_events, num_events / duration)
        p_empirical = np.zeros([ndim])
        for k in range(ndim):
            p_empirical[k] = np.sum(s == k+1)
        p_empirical /= p_empirical.sum()
        print 'True distribution: %s' % str(p_prior)
        print 'Empirical distribution: %s' % str(p_empirical)

        # convert the signal array into a binary one-of-k matrix for visualization
        B = np.zeros([ndim, nt], dtype='bool')
        for k in range(ndim):
            i = s == k+1
            B[k, i] = True

        plt.figure()
        plt.imshow(B.astype('int'), interpolation='nearest', aspect='auto', extent=[t.min(), t.max(), 0, ndim], cmap=plt.cm.bone_r)
        plt.show()

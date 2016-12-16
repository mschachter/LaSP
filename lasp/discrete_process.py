import numpy as np


class DiscreteProcessEstimator(object):
    """ This class estimates the emperical distribution of a discrete random variable over time. It forgets the
        distribution with a user-specified time constant.
    """

    def __init__(self, num_states, sample_rate, time_constant):
        self.num_states = num_states
        self.state = np.zeros([num_states])
        self.sample_rate = sample_rate
        self.p = np.zeros([num_states])
        self.tau = time_constant
        self.dt = 1. / self.sample_rate


    def update(self, new_observation):
        """ Update the state of the system with a new observation.

        :param new_observation: An integer from 1 to self.num_states, indicated the category (level) of the
               variable (factor). A zero indicates that there is no observation for this time period.

        :return:
        """

        # convert the observation to one-of-k encoding
        v = np.zeros([self.num_states])
        if new_observation > 0:
            v[new_observation-1] = 1.
        self.state = self.state * (1. - (self.dt / self.tau)) + v
        self.p = self.state / np.sum(self.state)

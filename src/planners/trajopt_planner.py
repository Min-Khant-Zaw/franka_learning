import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint


class TrajOpt(object):

    def __init__(self, n_waypoints, start, goal):
        """ set hyperparameters """
        self.n_waypoints = n_waypoints
        self.n_joints = len(start)
        self.start = start
        self.goal = goal
        """ create initial trajectory """
        self.xi0 = np.zeros((self.n_waypoints, self.n_joints))  # 5x7
        for idx in range(self.n_waypoints):
            self.xi0[idx,:] = self.start + idx/(self.n_waypoints - 1.0) * (self.goal - self.start)
        self.xi0 = self.xi0.reshape(-1) # flatten initial trajectory into 1D array for scipy minimize (1x35)
        """ create start point equality constraint """
        self.B = np.zeros((self.n_joints, self.n_joints * self.n_waypoints))    # 7x35
        for idx in range(self.n_joints):
            self.B[idx,idx] = 1
        self.start_constraint = LinearConstraint(self.B, self.start, self.start)
        # you can define a similar constraint for the end, if the final position is fixed

    """ problem specific cost function """
    def trajcost(self, xi):
        xi = xi.reshape((self.n_waypoints, self.n_joints))    # 5x7
        cost_total = 0
        for idx in range(self.n_waypoints):
            # state cost goes here
            cost_total += cost_state
        # perhaps you have some overall trajectory cost
        cost_total += cost_trajectory
        return cost_total

    """ use scipy optimizer to get optimal trajectory """
    def optimize(self, method='SLSQP'):
        start_t = time.time()
        res = minimize(self.trajcost, self.xi0, method=method, constraints=self.start_constraint)
        xi = res.x.reshape(self.n_waypoints,self.n_joints)
        return xi, res, time.time() - start_t
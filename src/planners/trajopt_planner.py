import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint


class TrajOpt(object):

    def __init__(self, n_waypoints, start, goal, feat_list, max_iter, environment):
        # Set hyperparameters
        self.n_joints = len(start)
        self.start = start
        self.goal = goal
        self.feat_list = feat_list
        self.num_features = len(self.feat_list)

        # Initilize weights for each feature
        self.weights = [0.0] * self.num_features

        # Variables for trajpot parameters
        self.n_waypoints = n_waypoints
        self.MAX_ITER = max_iter

        # Set Pybullet environment
        self.environment = environment

        # Create initial trajectory
        self.xi0 = np.zeros((self.n_waypoints, self.n_joints))  # 5x7
        for idx in range(self.n_waypoints):
            self.xi0[idx,:] = self.start + idx/(self.n_waypoints - 1.0) * (self.goal - self.start)
        self.xi0 = self.xi0.reshape(-1) # flatten initial trajectory into 1D array for scipy minimize (1x35)

        # Create start point equality constraint
        self.B = np.zeros((self.n_joints, self.n_joints * self.n_waypoints))    # 7x35
        for idx in range(self.n_joints):
            self.B[idx,idx] = 1
        self.start_constraint = LinearConstraint(self.B, self.start, self.start)
        # you can define a similar constraint for the end, if the final position is fixed

    # Cost functions
    def efficiency_cost(self, waypt):
        """
		Computes the total efficiency cost
		---
		input two consecutive waypoints, output scalar cost
		"""
        prev_waypt = waypt[0:7]
        current_waypt = waypt[7:14]
        feature = self.environment.efficiency_features(current_waypt, prev_waypt)
        feature_idx = self.feat_list.index("efficiency")
        return feature * self.weights[feature_idx]
    
    def origin_cost(self, waypt):
        """
		Computes the total distance from EE to base of robot cost.
		---
		input waypoint, output scalar cost
		"""
        feature = self.environment.origin_features(waypt)
        feature_idx = self.feat_list.index('origin')
        return feature * self.weights[feature_idx]
    
    def table_cost(self, waypt):
        """
        Computes the total distance to table cost.
        ---
		input waypoint, output scalar cost
        """
        feature = self.environment.table_features(waypt)
        feature_idx = self.feat_list.index("table")
        return feature * self.weights[feature_idx]
    
    def coffee_cost(self, waypt):
        """
        Computes the total coffee (EE orientation cost)
        ---
        input waypoint, output scalar cost
        """
        feature = self.environment.coffee_features(waypt)
        feature_idx = self.feat_list.index("coffee")
        return feature * self.weights[feature_idx]
    
    def laptop_cost(self, waypt):
        """
        Computes the total distance to laptop cost
		---
		input waypoint, output scalar cost
		"""
        prev_waypt = waypt[0:7]
        curr_waypt = waypt[7:14]
        feature = self.environment.laptop_features(curr_waypt,prev_waypt)
        feature_idx = self.feat_list.index('laptop')
        return feature * self.weights[feature_idx] * np.linalg.norm(curr_waypt - prev_waypt)

    def human_cost(self, waypt):
        """
		Computes the total distance to human cost.
		---
		input waypoint, output scalar cost
		"""
        prev_waypt = waypt[0:7]
        curr_waypt = waypt[7:14]
        feature = self.environment.human_features(curr_waypt,prev_waypt)
        feature_idx = self.feat_list.index('human')
        return feature * self.weights[feature_idx] * np.linalg.norm(curr_waypt - prev_waypt)

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
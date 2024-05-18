import math
import numpy as np 
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pdb
import sys

from ilqr.vehicle_model import Model
from ilqr.local_planner import LocalPlanner
from ilqr.constraints import Constraints


class multiiLQR():
    def __init__(self, args, obstacle_bb):
        self.args = args
        self.Ts = args.timestep
        self.N = args.horizon
        self.obstacle_bb = obstacle_bb
        
        self.global_plan = None
        self.local_planner = LocalPlanner(args)
        self.vehicle_model = Model(args)
        self.constraints = Constraints(args, obstacle_bb)
        
        # initial nominal trajectory
        self.control_seq = np.zeros((self.args.num_ctrls, self.args.horizon))
        self.control_seq[0, :] = np.ones((self.args.horizon)) * 0.5

        self.lamb_factor = 10
        self.max_lamb = 1000

    
    def set_global_plan(self, global_plan):
        self.global_plan = global_plan
        self.local_planner.set_global_planner(self.global_plan)

    def get_nominal_trajectory(self, X_0, U):
        X = np.zeros((self.args.num_states, self.args.horizon+1))
        X[:, 0] = X_0
        for i in range(self.args.horizon):
            X[:, i+1] = self.vehicle_model.forward_simulate(X[:, i], U[:, i])
        return X

    def forward_pass(self, X, U, d, K):
        X_new = np.zeros((self.args.num_states, self.args.horizon+1))
        X_new[:, 0] = X[:, 0]
        U_new = np.zeros((self.args.num_ctrls, self.args.horizon))
        # Do a forward rollout and get states at all control points
        for i in range(self.args.horizon):
            U_new[:, i] = U[:, i] + d[:, i] + K[:, :, i] @ (X_new[:, i] - X[:, i])
            X_new[:, i+1] = self.vehicle_model.forward_simulate(X_new[:, i], U_new[:, i])
        return X_new, U_new

    def backward_pass(self, X, U, poly_coeff, x_local_plan, npc_traj, lamb):
        # Find control sequence that minimizes Q-value function
        # Get derivatives of Q-function wrt to state and control
        l_x, l_xx, l_u, l_uu, l_ux = self.constraints.get_cost_derivatives(X[:, 1:], U, poly_coeff, x_local_plan, npc_traj) 
        df_dx = self.vehicle_model.get_A_matrix(X[2, 1:], X[3, 1:], U[0,:])
        df_du = self.vehicle_model.get_B_matrix(X[3, 1:])
        # Value function at final timestep is known
        V_x = l_x[:,-1] 
        V_xx = l_xx[:,:,-1]
        # Allocate space for feedforward and feeback term
        d = np.zeros((self.args.num_ctrls, self.args.horizon))
        K = np.zeros((self.args.num_ctrls, self.args.num_states, self.args.horizon))
        # Run a backwards pass from N-1 control step
        for i in range(self.args.horizon-1,-1,-1):
            Q_x = l_x[:,i] + df_dx[:,:,i].T @ V_x
            Q_u = l_u[:,i] + df_du[:,:,i].T @ V_x
            Q_xx = l_xx[:,:,i] + df_dx[:,:,i].T @ V_xx @ df_dx[:,:,i] 
            Q_ux = l_ux[:,:,i] + df_du[:,:,i].T @ V_xx @ df_dx[:,:,i]
            Q_uu = l_uu[:,:,i] + df_du[:,:,i].T @ V_xx @ df_du[:,:,i]
            # Q_uu_inv = np.linalg.pinv(Q_uu)
            Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
            Q_uu_evals[Q_uu_evals < 0] = 0.0
            Q_uu_evals += lamb
            Q_uu_inv = np.dot(Q_uu_evecs,np.dot(np.diag(1.0/Q_uu_evals), Q_uu_evecs.T))

            # Calculate feedforward and feedback terms
            d[:,i] = -Q_uu_inv @ Q_u
            K[:,:,i] = -Q_uu_inv @ Q_ux
            # Update value function for next time step
            V_x = Q_x - K[:,:,i].T @ Q_uu @ d[:,i]
            V_xx = Q_xx - K[:,:,i].T @ Q_uu @ K[:,:,i]
        
        return d, K

    def get_optimal_control_seq(self, X_0, U, poly_coeff, x_local_plan, npc_traj):
        X = self.get_nominal_trajectory(X_0, U)
        J_old = sys.float_info.max
        lamb = 1 # Regularization parameter
        # Run iLQR for max iterations
        for itr in range(self.args.max_iters):
            d, K = self.backward_pass(X, U, poly_coeff, x_local_plan, npc_traj, lamb)
            # Get control values at control points and new states again by a forward rollout
            X_new, U_new = self.forward_pass(X, U, d, K)
            J_new = self.constraints.get_total_cost(X, U, poly_coeff, x_local_plan)
            
            if J_new < J_old:
                X = X_new
                U = U_new
                lamb /= self.lamb_factor
                if (abs(J_old - J_new) < self.args.tol):
                    # print("Tolerance reached")
                    break
            else:
                lamb *= self.lamb_factor
                if lamb > self.max_lamb:
                    break
            
            J_old = J_new
        # print(J_new)
        return X, U
    
    def run_step(self, agent_state, npc_traj):
        assert self.global_plan is not None, "Set a global plan in iLQR before starting run_step"

        self.local_planner.set_agent_state(agent_state)
        ref_traj, poly_coeff = self.local_planner.get_local_plan()

        X_0 = np.array([agent_state[0], agent_state[1], agent_state[2], agent_state[3]])

        X, U = self.get_optimal_control_seq(X_0, self.control_seq, poly_coeff, ref_traj[:, 0], npc_traj)

        self.control_seq = U
        return X, U 

import math
import numpy as np 
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pdb
import sys
import warnings

from ilqr.vehicle_model import BicycleModel, QuadcopterModel
from ilqr.local_planner import LocalPlanner
from ilqr.constraints import Constraints, SAMVGaussian
from collections import deque

np.seterr(over='ignore', invalid='ignore')

def compute_precision(mu, Sigma, Dx, hori, reg=1e-2):
    horizon = mu.shape[0]
    
    # print(horizon)
    # obtain the reference precision
    Delta_t = np.zeros((hori+1, Dx, Dx))
    if hori < horizon:
        #if the horizon is within the remaining time steps, extract the marginal distribution
        for i in range(hori):
            Delta_t[i+1] = np.linalg.pinv(Sigma[Dx*i:Dx*(i+1), Dx*i:Dx*(i+1)]+ reg*np.eye(Dx))
    else:
        #if the horizon exceeds the remaining time steps
        for i in range(horizon):
            Delta_t[i+1] = np.linalg.pinv(Sigma[Dx*i:Dx*(i+1), Dx*i:Dx*(i+1)]+ reg*np.eye(Dx))
        for i in range(horizon, hori):
            Delta_t[i+1] = Delta_t[horizon]

    Delta_t[0] = Delta_t[1].copy()
    return Delta_t

class iLQR():
    def __init__(self, args, obstacle_bb, agent_ID="1"):
        self.args = args
        self.Ts = args.timestep
        self.obstacle_bb = np.array(obstacle_bb)
        self.agent_ID = agent_ID
        self.global_plan = None
        self.Dx = args.num_states
        self.Du = args.num_ctrls

        self.local_planner = LocalPlanner(args, agent_ID)
        self.vehicle_model = QuadcopterModel(args)
        self.constraints = Constraints(args, obstacle_bb)
        
        self.Qs = np.zeros(((args.horizon_short+1)*self.Dx,(args.horizon_short+1)*self.Dx))
        self.Rs = np.zeros(((args.horizon_short+1)*self.Du,(args.horizon_short+1)*self.Du))

        self.tmp_lx = deque(maxlen=20)
        self.tmp_lxx = deque(maxlen=20)
        self.tmp_lu = deque(maxlen=20)
        self.tmp_luu = deque(maxlen=20)
        self.tmp_lux = deque(maxlen=20)
        self.tmp_fx = deque(maxlen=20)
        self.tmp_fu = deque(maxlen=20)
        self.total_lx = []; self.total_lxx = []; self.total_lu = []; 
        self.total_luu = []; self.total_lux = []; self.total_fx = []; self.total_fu = []
        # initial nominal trajectory
        self.control_seq_long = np.zeros((self.args.num_ctrls, self.args.horizon))
        self.control_seq_long[0, :] = np.ones((self.args.horizon)) * 1
        # print(self.control_seq_long.shape)
        self.total_cost = 0

        self.lamb_factor = 10
        self.max_lamb = 1000

    
    def set_global_plan(self, global_plan):
        self.global_plan = global_plan
        self.local_planner.set_global_planner(self.global_plan)

    def get_nominal_trajectory(self, X_0, U):
        # print(X_0.shape, U.shape)
        X = np.zeros((self.args.num_states, self.args.horizon+1))
        # print(X_0.shape, X.shape)
        X[:, 0] = X_0
        for i in range(self.args.horizon):
            # print(X[:, i].shape, self.vehicle_model.forward_simulate(X[:, i], U[:, i]).shape)
            X[:, i+1] = self.vehicle_model.forward_simulate(X[:, i], U[:, i])
        return X
    
    def get_nominal_trajectory_short(self, X_0, U):
        # print(X_0.shape, U.shape)
        X = np.zeros((self.args.num_states, self.args.horizon_short+1))
        X[:, 0] = X_0
        for i in range(self.args.horizon_short):
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
            # print(U_new[:, i])
        # print(X_new[:,0])

        return X_new, U_new
    
    def forward_pass_short(self, X, U, d, K):
        X_new = np.zeros((self.args.num_states, self.args.horizon_short+1))
        X_new[:, 0] = X[:, 0]
        U_new = np.zeros((self.args.num_ctrls, self.args.horizon_short))
        # Do a forward rollout and get states at all control points
        for i in range(self.args.horizon_short):
            U_new[:, i] = U[:, i] + d[:, i] + K[:, :, i] @ (X_new[:, i] - X[:, i])
            X_new[:, i+1] = self.vehicle_model.forward_simulate(X_new[:, i], U_new[:, i])

        return X_new, U_new

    def backward_pass(self, X, U, poly_coeff_y, poly_coeff_z, x_local_plan, npc_traj, lamb, current_epoch):
        # Find control sequence that minimizes Q-value function
        # Get derivatives of Q-function wrt to state and control
        # print(X.shape, X[:, 1:].shape)
        l_x, l_xx, l_u, l_uu, l_ux = self.constraints.get_cost_derivatives_long(X[:, 1:], U, poly_coeff_y, poly_coeff_z, x_local_plan, npc_traj, current_epoch) 
        # print(l_x.shape, l_xx.shape, l_u.shape, l_uu.shape, l_ux.shape)
        # print(l_u)
        df_dx = self.vehicle_model.get_A_matrix(X[:, 1:],  U[:, :], self.args.horizon)
        df_du = self.vehicle_model.get_B_matrix(X[:, 1:],  U[:, :], self.args.horizon)
        # print(X[2, 1:].shape)
        # Value function at final timestep is known
        V_x = l_x[:,-1] # 4x1
        V_xx = l_xx[:,:,-1] # 4x4
        # print(V_x.shape, V_xx.shape)
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
            # print(l_uu)
            # Q_uu_inv = np.linalg.pinv(Q_uu)
            Q_uu = self.constraints.clip_derivatives(Q_uu)
            Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
            Q_uu_evals[Q_uu_evals < 0] = 0.0
            Q_uu_evals += lamb
            Q_uu_inv = self.constraints.clip_derivatives(np.dot(Q_uu_evecs,np.dot(np.diag(1.0/Q_uu_evals), Q_uu_evecs.T)))
            # Calculate feedforward and feedback terms
            d[:,i] = -Q_uu_inv @ Q_u
            K[:,:,i] = -Q_uu_inv @ Q_ux
            # Update value function for next time step
            V_x = Q_x - K[:,:,i].T @ Q_uu @ d[:,i]
            V_xx = Q_xx - K[:,:,i].T @ Q_uu @ K[:,:,i]
        return d, K
    
    def backward_pass_short(self, X, U, ref_traj, npc_traj, lamb, t):
        # Find control sequence that minimizes Q-value function
        # Get derivatives of Q-function wrt to state and control
        # print(X.shape, X[:, 1:].shape)
        l_x, l_xx, l_u, l_uu, l_ux = self.constraints.get_cost_derivatives_short(X[:, 1:], U, ref_traj, npc_traj, t) 
        # print(l_x.shape, l_xx.shape, l_u.shape, l_uu.shape, l_ux.shape)
        # print(X.shape, U.shape)
        df_dx = self.vehicle_model.get_A_matrix(X[:, 1:],  U[:, :], self.args.horizon_short)
        df_du = self.vehicle_model.get_B_matrix(X[:, 1:],  U[:, :], self.args.horizon_short)

        # Value function at final timestep is known
        V_x = l_x[:,-1] # 4x1
        V_xx = l_xx[:,:,-1] # 4x4
        # print(V_x.shape, V_xx.shape)
        # Allocate space for feedforward and feeback term
        d = np.zeros((self.args.num_ctrls, self.args.horizon_short))
        K = np.zeros((self.args.num_ctrls, self.args.num_states, self.args.horizon_short))
        # Run a backwards pass from N-1 control step
        for i in range(self.args.horizon_short-1,-1,-1):
            Q_x = l_x[:,i] + df_dx[:,:,i].T @ V_x
            Q_u = l_u[:,i] + df_du[:,:,i].T @ V_x
            Q_xx = l_xx[:,:,i] + df_dx[:,:,i].T @ V_xx @ df_dx[:,:,i] 
            Q_ux = l_ux[:,:,i] + df_du[:,:,i].T @ V_xx @ df_dx[:,:,i]
            Q_uu = l_uu[:,:,i] + df_du[:,:,i].T @ V_xx @ df_du[:,:,i]
            Q_uu = self.constraints.clip_derivatives(Q_uu)
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

    def get_optimal_control_seq_test(self, X_0, U, poly_coeff_y, poly_coeff_z, x_local_plan, npc_traj, id):
        X = self.get_nominal_trajectory(X_0, U)
        # print(X)
        J_old = sys.float_info.max
        lamb = 1 # Regularization parameter
        # Run iLQR for max iterations
        for itr in range(self.args.max_iters):
            # print(poly_coeff_y, poly_coeff_z)
            d, K = self.backward_pass(X, U, poly_coeff_y, poly_coeff_z, x_local_plan, npc_traj, lamb, current_epoch)
            # Get control values at control points and new states again by a forward rollout
            X_new, U_new = self.forward_pass(X, U, d, K)
            # print(U_new)
            # print(X_new)
            # print(id)
            J_new = self.constraints.get_total_cost_test(X, U, poly_coeff_y, poly_coeff_z, x_local_plan, id)
            # print(itr)
            # print(J_new)
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
        # print(X_new)
        return X, U, J_new
    
    def get_optimal_control_seq(self, X_0, U, poly_coeff_y, poly_coeff_z, x_local_plan, npc_traj, current_epoch):
        X = self.get_nominal_trajectory(X_0, U)
        # print(X)
        J_old = sys.float_info.max
        lamb = 1 # Regularization parameter
        # Run iLQR for max iterations
        for itr in range(self.args.max_iters):
            # print(poly_coeff_y, poly_coeff_z)
            d, K = self.backward_pass(X, U, poly_coeff_y, poly_coeff_z, x_local_plan, npc_traj, lamb, current_epoch)
            # Get control values at control points and new states again by a forward rollout
            X_new, U_new = self.forward_pass(X, U, d, K)
            # print(U_new)
            # print(X_new)

            J_new = self.constraints.get_total_cost(X, U, poly_coeff_y, poly_coeff_z, x_local_plan)
            # print(itr)
            # print(J_new)
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
        # print(X_new)
        return X, U, J_new

    def get_optimal_control_seq_short(self, X_0, U, ref_traj, t):
        # X = self.get_nominal_trajectory_short(X_0, U)
        X = np.concatenate((X_0[:, np.newaxis], ref_traj), axis=1)
        # print(X.shape, X_0.shape, ref_traj.shape)
        J_old = sys.float_info.max
        lamb = 1 # Regularization parameter
        # Run iLQR for max iterations
        for itr in range(self.args.max_iters):
            d, K = self.backward_pass_short(X, U, ref_traj, [], lamb, t)

            # Get control values at control points and new states again by a forward rollout
            X_new, U_new = self.forward_pass_short(X, U, d, K)

            J_new = self.constraints.get_total_cost_short(X, U, ref_traj, t)
            # print(itr)
            # print(J_new)
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
        return X, U, J_new

    def run_step_long(self, agent_state, npc_traj, current_epoch):
        assert self.global_plan is not None, "Set a global plan in iLQR before starting run_step"
        # assert len(agent_state) is 12, "agent_state should be of shape (12,)"
        self.local_planner.set_agent_state(agent_state)
        ref_traj, poly_coeff_y, poly_coeff_z = self.local_planner.get_local_plan()
        X_0 = agent_state
        # print("ref=", ref_traj[:, 0])
        # print("X0=", X_0)
        # print(self.agent_ID)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=np.ComplexWarning)
            X, U, total_cost = self.get_optimal_control_seq(X_0, self.control_seq_long, poly_coeff_y, poly_coeff_z, ref_traj[:, 0], npc_traj, current_epoch)
        self.control_seq_long = U
        return X, U, total_cost

    def run_step_short_marginal(self, agent_state, nom_traj, nom_ctrl, t):
        X_0 = agent_state
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=np.ComplexWarning)
            X, U, total_cost = self.get_optimal_control_seq_short(X_0, nom_ctrl[:, :self.args.horizon_short], 
                                                              nom_traj[:, t:t+self.args.horizon_short], t)

        # self.control_seq_long = U
        return X, U, total_cost
    
    def run_step_short_conditional(self, agent_state, nom_traj, nom_ctrl, t):
        X_0 = agent_state
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=np.ComplexWarning)
            X, U, total_cost = self.get_optimal_control_seq_short(X_0, nom_ctrl[:, :self.args.horizon_short], 
                                                              nom_traj[:, :self.args.horizon_short], t)

        # self.control_seq_long = U
        return X, U, total_cost
    
    def compute_nom_traj_distribution(self, l_x, l_xx, l_u, l_uu, l_ux, reg=1e-2):
        hori = self.args.horizon
        # print(np.array(l_x).shape, np.array(l_xx).shape, np.array(l_u).shape, np.array(l_uu).shape, np.array(l_ux).shape)
        self.Qs = np.zeros((hori*self.Dx,hori*self.Dx))
        self.Rs = np.zeros((hori*self.Du,hori*self.Du))
        # print("len(lxx)=", len(l_xx))
        for i in range(hori):
            self.Qs[self.Dx*i:self.Dx*(i+1),self.Dx*i:self.Dx*(i+1)] = l_xx[i]
            self.Rs[self.Du*i:self.Du*(i+1),self.Du*i:self.Du*(i+1)] = l_uu[i]
        self.Sx = np.zeros((self.Dx*hori,self.Dx))
        self.Su = np.zeros((self.Dx*hori,self.Du*hori))  
        # print(self.Sx[self.Dx*(i-1):self.Dx*(i), :].shape)
        
        self.Sx[0:self.Dx, :] = np.eye(self.Dx)
        for i in range(1, hori):
            self.Sx[self.Dx*i:self.Dx*(i+1), :] =  (self.Sx[self.Dx*(i-1):self.Dx*(i), :]) @ (self.tmp_fx[i-1])
        for i in range(1, hori):
            self.Su[self.Dx*i:self.Dx*(i+1), self.Du*(i-1): self.Du*(i)] = self.tmp_fu[i-1]
            self.Su[self.Dx*i:self.Dx*(i+1), :self.Du*(i-1)] = (self.tmp_fx[i-1]) @ (self.Su[self.Dx*(i-1):self.Dx*(i), :self.Du*(i-1)])

        lx_flat = np.array(l_x).flatten()
        lu_flat = np.array(l_u).flatten()
        # print("lx, lu:", lx_flat.shape, lu_flat.shape)

        Sigma_u_inv = (self.Su.T @ self.Qs @ self.Su) + self.Rs
        delta_us = -np.linalg.solve(Sigma_u_inv, self.Su.T @ self.Qs @ self.Sx @ -np.zeros(self.Dx))+ lx_flat @ self.Su + lu_flat
        delta_xs = self.Sx @ np.zeros(self.Dx) + self.Su @ delta_us
        Sigma_u = np.linalg.inv(Sigma_u_inv)
        Sigma_x = self.Su @ Sigma_u @ self.Su.T 
        # print(Sigma_u_inv.shape, delta_us.shape, delta_xs.shape)
        # print(Sigma_u.shape, Sigma_x.shape)

        return delta_us, delta_xs, Sigma_u_inv, Sigma_u, Sigma_x

    def compute_nom_npc_traj_distribution(self, l_x, l_xx, l_u, l_uu, l_ux, c_x, c_xx, reg=1e-2):
        hori = self.args.horizon
        c_mag = 1e2
        # print(np.array(l_x).shape, np.array(l_xx).shape, np.array(l_u).shape, np.array(l_uu).shape, np.array(l_ux).shape)
        self.Qs = np.zeros((hori*self.Dx,hori*self.Dx))
        self.Rs = np.zeros((hori*self.Du,hori*self.Du))
        self.Cx = np.zeros((self.Dx*hori, 1))
        self.Cxx = np.zeros((hori*self.Dx,hori*self.Dx))
        # print("len(lxx)=", len(l_xx))
        for i in range(hori):
            self.Qs[self.Dx*i:self.Dx*(i+1),self.Dx*i:self.Dx*(i+1)] = l_xx[i]
            self.Rs[self.Du*i:self.Du*(i+1),self.Du*i:self.Du*(i+1)] = l_uu[i]
        for i in range(hori):
            self.Cxx[self.Dx*i:self.Dx*(i+1),self.Dx*i:self.Dx*(i+1)] = c_xx[i]
        self.Sx = np.zeros((self.Dx*hori,self.Dx))
        self.Su = np.zeros((self.Dx*hori,self.Du*hori))  
        # print(self.Sx[self.Dx*(i-1):self.Dx*(i), :].shape)

        self.Sx[0:self.Dx, :] = np.eye(self.Dx)
        for i in range(1, hori):
            self.Sx[self.Dx*i:self.Dx*(i+1), :] =  (self.Sx[self.Dx*(i-1):self.Dx*(i), :]) @ (self.tmp_fx[i-1])
        for i in range(1, hori):
            self.Su[self.Dx*i:self.Dx*(i+1), self.Du*(i-1): self.Du*(i)] = self.tmp_fu[i-1]
            self.Su[self.Dx*i:self.Dx*(i+1), :self.Du*(i-1)] = (self.tmp_fx[i-1]) @ (self.Su[self.Dx*(i-1):self.Dx*(i), :self.Du*(i-1)])
        for i in range(hori):
            self.Cx[self.Dx*i:self.Dx*(i+1), :] = c_x[i].reshape((-1, 1))

        lx_flat = np.array(l_x).flatten()
        lu_flat = np.array(l_u).flatten()
        # print("lx, lu:", lx_flat.shape, lu_flat.shape)
        # print((self.Su.T @ self.Cx).shape)
        Sigma_u_inv = (self.Su.T @ (self.Qs+self.Cxx*c_mag) @ self.Su) + self.Rs
        delta_us = -np.linalg.solve(Sigma_u_inv, self.Su.T @ (self.Qs+self.Cxx*c_mag) @ self.Sx @ -np.zeros(self.Dx) 
                                    + 0.5 * (self.Su.T @ self.Cx*c_mag).flatten())+ lx_flat @ self.Su + lu_flat
        delta_xs = self.Sx @ np.zeros(self.Dx) + self.Su @ delta_us
        Sigma_u = np.linalg.inv(Sigma_u_inv)
        Sigma_x = self.Su @ Sigma_u @ self.Su.T 
        # print(self.Cx.shape, self.Su.shape, (self.Su.T @ (self.Qs+self.Cxx) @ self.Sx @ -np.zeros(self.Dx)).shape)
        # print(Sigma_u_inv.shape, delta_us.shape, delta_xs.shape)
        # print(Sigma_u.shape, Sigma_x.shape)

        return delta_us, delta_xs, Sigma_u_inv, Sigma_u, Sigma_x

    def get_state_cost_marginal(self, Sigma_x, nom_traj):
        Sigma_x = self.constraints.clip_derivatives(Sigma_x)
        Delta_t = compute_precision(nom_traj.T, Sigma_x, self.args.num_states, self.args.horizon)
        return Delta_t
    
    def get_state_cost_single_conditional(self, Sigma_x, nom_traj, flag_terminal):
        # print(nom_traj.shape)
        Sigma_x = self.constraints.clip_derivatives(Sigma_x)
        Delta_t = np.zeros((self.args.horizon_short+1, self.args.num_states, self.args.num_states))
        nom_traj = nom_traj.T

        x_distrib_long = SAMVGaussian()
        x_distrib_long.D = (self.args.horizon+1)*self.args.num_states
        x_distrib_long.mu = nom_traj.flatten()
        x_distrib_long.Sigma = Sigma_x
        # print("x_distrib=", x_distrib_long.mu.shape, x_distrib_long.Sigma.shape)
        x_distrib_short = SAMVGaussian()
        x_distrib_short.D = (self.args.horizon_short+1)*self.args.num_states # 4*(10+1)

        for t in range(self.args.horizon_short):
            mu_marg, sigma_marg = x_distrib_long.get_marginal_distrib(slice(t*self.args.num_states, (t+1+self.args.horizon_short)*self.args.num_states))
            x_distrib_short.mu = mu_marg
            x_distrib_short.Sigma = sigma_marg

            mu_cond, sigma_cond = x_distrib_short.get_condition_distrib(nom_traj[t, :], 
                                            slice(0, self.args.num_states), slice(self.args.num_states, (self.args.horizon_short+1)*self.args.num_states))
            Delta_t = compute_precision(mu_cond, sigma_cond, self.args.num_states, self.args.horizon_short)


        return Delta_t


        # self.constraints.state_cost_t = Delta_t[1:]

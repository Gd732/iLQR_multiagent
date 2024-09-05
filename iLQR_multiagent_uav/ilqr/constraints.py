import numpy as np 

class Constraints:
	def __init__(self, args, obstacle_bb):
		self.args = args
		self.state_cost_c = np.diag((self.args.w_pos, self.args.w_pos, self.args.w_pos, self.args.w_vel, self.args.w_vel, self.args.w_vel, 0, 0, 0, 0, 0, 0))
		self.state_cost_c_termin = self.state_cost_c * 5
		# self.control_cost_c = np.array([[self.args.w_acc,                   0],
		# 							  [              0, self.args.w_yawrate]]) # R
		self.control_cost_c = np.diag((1, 1, 1, 1))
		self.state_cost_t = np.repeat(self.state_cost_c[np.newaxis, :, :], args.horizon_short, axis=0)
		self.total_state_cost_t = []
		self.coeffs = None

		# self.number_of_npc = 1 # hardcode
   
		self.r = args.drone_radius

	def get_state_cost_derivatives_long(self, state, poly_coeffs_y, poly_coeffs_z, x_local_plan, npc_traj, current_epoch):
		"""
		Returns the first order and second order derivative of the value function wrt state
		"""
		l_x = np.zeros((self.args.num_states, self.args.horizon))
		l_xx = np.zeros((self.args.num_states, self.args.num_states, self.args.horizon))
		for i in range(self.args.horizon):
			# Offset in path derivative
			x_r, y_r, z_r = self.find_closest_point_3d(state[:, i], poly_coeffs_y, poly_coeffs_z, x_local_plan)
			# print(x_r)
			# print(state[:, i], x_r, y_r)
			if current_epoch >= self.args.epochs - self.args.horizon:
				traj_cost = 2*self.state_cost_c_termin@(np.array([state[0, i]-x_r, state[1, i]-y_r, state[2, i]-z_r, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
				l_xx_i = 2*self.state_cost_c_termin
			else:
				traj_cost = 2*self.state_cost_c@(np.array([state[0, i]-x_r, state[1, i]-y_r, state[2, i]-z_r, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
				l_xx_i = 2*self.state_cost_c

			# Compute first order derivative
			l_x_i = traj_cost
			# Compute second order derivative
			# Obstacle derivative
			l_b_dot_obs = []
			l_b_ddot_obs = []
			for k in range(len(npc_traj)):
				# print(len(npc_traj))
				# print(npc_traj.shape)
				npc_traj_k = np.squeeze(npc_traj[k], axis=0) if len(npc_traj[k].shape) == 3 else npc_traj[k]
				# print(npc_traj_k.shape)
				b_dot_obs_k, b_ddot_obs_k = self.compute_obstacle_cost_derivatives(npc_traj_k, i, state[:, i])
				l_b_dot_obs.append(b_dot_obs_k)
				l_b_ddot_obs.append(b_ddot_obs_k)
    
			b_dot_obs = np.sum(l_b_dot_obs, axis=0)
			b_ddot_obs = np.sum(l_b_ddot_obs, axis=0)
			# print(b_dot_obs)
			l_x_i += b_dot_obs.squeeze()
			l_xx_i += b_ddot_obs
    
			l_xx[:, :, i] = l_xx_i
			l_x[:, i] = l_x_i

		return l_x, l_xx

	def get_state_cost_derivatives_short(self, state, ref_traj, npc_traj, t):
		"""
		Returns the first order and second order derivative of the value function wrt state
		"""
		l_x = np.zeros((self.args.num_states, self.args.horizon_short))
		l_xx = np.zeros((self.args.num_states, self.args.num_states, self.args.horizon_short))
		for i in range(self.args.horizon_short):
			# Offset in path derivative
			x_r, y_r, z_r = ref_traj[0, i], ref_traj[1, i], ref_traj[2, i]
			vx_r, vy_r, vz_r = ref_traj[3, i], ref_traj[4, i], ref_traj[5, i]
			traj_cost = 2*self.state_cost_t[i+t]@(np.array([state[0, i]-x_r, state[1, i]-y_r, state[2, i]-z_r, 
												   state[3, i]-vx_r, state[4, i]-vy_r, state[5, i]-vz_r, 0, 0, 0, 0, 0, 0]))
			# print(self.state_cost_t[i])
			# Compute first order derivative
			l_x_i = traj_cost
			# Compute second order derivative
			l_xx_i = 2*self.state_cost_t[i+t]
			# Obstacle derivative
			l_b_dot_obs = []
			l_b_ddot_obs = []
			# for k in range(len(npc_traj)):
			# 	# print(len(npc_traj))
			# 	npc_traj_k = np.squeeze(npc_traj[k], axis=0) if len(npc_traj[k].shape) == 3 else npc_traj[k]
    
			# 	b_dot_obs_k, b_ddot_obs_k = self.get_obstacle_cost_derivatives_short(npc_traj_k, i, state[:, i])
			# 	l_b_dot_obs.append(b_dot_obs_k)
			# 	l_b_ddot_obs.append(b_ddot_obs_k)
    
			b_dot_obs = np.sum(l_b_dot_obs, axis=0)
			b_ddot_obs = np.sum(l_b_ddot_obs, axis=0)
   
			l_x_i += b_dot_obs.squeeze()
			l_xx_i += b_ddot_obs
    
			l_xx[:, :, i] = l_xx_i
			l_x[:, i] = l_x_i
		return l_x, l_xx


	def get_control_cost_derivatives(self, state, control, horizon):
		"""
		Returns the control quadratic (R matrix) and linear cost term (r vector) for the trajectory
		"""
		P1 = np.array([[1],[0],[0],[0]])
		P2 = np.array([[0],[1],[0],[0]])
		P3 = np.array([[0],[0],[1],[0]])
		P4 = np.array([[0],[0],[0],[1]])

		l_u = np.zeros((self.args.num_ctrls, horizon))
		l_uu = np.zeros((self.args.num_ctrls, self.args.num_ctrls, horizon))
		# c_ctrl = 0
		for i in range(horizon):
			c1 = (np.matmul(control[:, i].T, P1) - self.args.acc_limits[1])
			b_1, b_dot_1, b_ddot_1 = self.barrier_function(self.args.q1_acc, self.args.q2_acc, c1, P1)
			c2 = (self.args.acc_limits[0] - np.matmul(control[:, i].T, P1))
			b_2, b_dot_2, b_ddot_2 = self.barrier_function(self.args.q1_acc, self.args.q2_acc, c2, -P1)

			c3 = (np.matmul(control[:, i].T, P2) - self.args.acc_limits[1])
			b_3, b_dot_3, b_ddot_3 = self.barrier_function(self.args.q1_acc, self.args.q2_acc, c3, P2)
			c4 = (self.args.acc_limits[0] - np.matmul(control[:, i].T, P2))
			b_4, b_dot_4, b_ddot_4 = self.barrier_function(self.args.q1_acc, self.args.q2_acc, c4, -P2)

			c5 = (np.matmul(control[:, i].T, P3) - self.args.acc_limits[1])
			b_5, b_dot_5, b_ddot_5 = self.barrier_function(self.args.q1_acc, self.args.q2_acc, c5, P3)
			c6 = (self.args.acc_limits[0] - np.matmul(control[:, i].T, P3))
			b_6, b_dot_6, b_ddot_6 = self.barrier_function(self.args.q1_acc, self.args.q2_acc, c6, -P3)

			c7 = (np.matmul(control[:, i].T, P4) - self.args.acc_limits[1])
			b_7, b_dot_7, b_ddot_7 = self.barrier_function(self.args.q1_acc, self.args.q2_acc, c7, P4)
			c8 = (self.args.acc_limits[0] - np.matmul(control[:, i].T, P4))
			b_8, b_dot_8, b_ddot_8 = self.barrier_function(self.args.q1_acc, self.args.q2_acc, c8, -P4)
			# print(c1, c2, c3, c4, c5, c6, c7, c8)
			l_u_i = b_dot_1 + b_dot_2 + b_dot_3 + b_dot_4 + b_dot_5 + b_dot_6 + b_dot_7 + b_dot_8 + (2*control[:, i].T @ self.control_cost_c).reshape(-1, 1)
			l_uu_i = b_ddot_1 + b_ddot_2 + b_ddot_3 + b_ddot_4 + b_ddot_5 + b_ddot_6 + b_ddot_7 + b_ddot_8 + 2*self.control_cost_c

			l_u[:, i] = l_u_i.squeeze()
			l_uu[:, :, i] = l_uu_i.squeeze()
		return l_u, l_uu

	def get_control_cost_derivatives_short(self, state, control, t):
		"""
		Returns the control quadratic (R matrix) and linear cost term (r vector) for the trajectory
		"""
		P1 = np.array([[1],[0]])
		P2 = np.array([[0],[1]])

		l_u = np.zeros((self.args.num_ctrls, self.args.horizon_short))
		l_uu = np.zeros((self.args.num_ctrls, self.args.num_ctrls, self.args.horizon_short))
		# c_ctrl = 0
		# print(state.shape, control.shape)
		for i in range(self.args.horizon_short):
			# Acceleration Barrier Max
			c = (np.matmul(control[:, i].T, P1) - self.args.acc_limits[1])
			b_1, b_dot_1, b_ddot_1 = self.barrier_function(self.args.q1_acc, self.args.q2_acc, c, P1)

			# Acceleration Barrier Min
			c = (self.args.acc_limits[0] - np.matmul(control[:, i].T, P1))
			b_2, b_dot_2, b_ddot_2 = self.barrier_function(self.args.q1_acc, self.args.q2_acc, c, -P1)

			velocity = state[2, i]

			# Yawrate Barrier Max
			c = self.args.w_yaw*(np.matmul(control[:, i].T, P2) - velocity*np.tan(self.args.steer_angle_limits[1])/self.args.wheelbase)
			b_3, b_dot_3, b_ddot_3 = self.barrier_function(self.args.q1_yawrate, self.args.q2_yawrate, c, P2)

			# Yawrate Barrier Min
			c = self.args.w_yaw*(velocity*np.tan(self.args.steer_angle_limits[0])/self.args.wheelbase - np.matmul(control[:, i].T, P2))
			b_4, b_dot_4, b_ddot_4 = self.barrier_function(self.args.q1_yawrate, self.args.q2_yawrate, c, -P2)

			l_u_i = b_dot_1 + b_dot_2 + b_dot_3 + b_dot_4 + (2*control[:, i].T @ self.control_cost_c).reshape(-1, 1)
			l_uu_i = b_ddot_1 + b_ddot_2 + b_ddot_3 + b_ddot_4 + 2*self.control_cost_c

			l_u[:, i] = l_u_i.squeeze()
			l_uu[:, :, i] = l_uu_i.squeeze()

		return l_u, l_uu


	def barrier_function(self, q1, q2, c, c_dot):
		# if c > 0 or c == np.nan:
		# 	print(c)
		c = 50 if c > 50 or c == np.nan else c
		b = q1*np.exp(q2*c)
		b_dot = q1*q2*np.exp(q2*c)*c_dot
		b_ddot = q1*(q2**2)*np.exp(q2*c)*np.matmul(c_dot, c_dot.T)

		return b, b_dot, b_ddot

	def get_cost_derivatives_long(self, state, control, poly_coeffs_y, poly_coeffs_z, x_local_plan, npc_traj, current_epoch):
		"""
		Returns the different cost terms for the trajectory
		This is the main function which calls all the other functions 
		"""
		# print(state.shape, control.shape)
		l_u, l_uu = self.get_control_cost_derivatives(state, control, self.args.horizon)
		# l_u, l_uu = np.zeros((4, 20)), np.zeros((4, 4, 20))
		l_x, l_xx = self.get_state_cost_derivatives_long(state, poly_coeffs_y, poly_coeffs_z, x_local_plan, npc_traj, current_epoch)
		l_ux = np.zeros((self.args.num_ctrls, self.args.num_states, self.args.horizon))
		# 4, 20 | 4, 4, 20 | 12, 20 | 12, 12, 20 | 4, 12, 20
		# print(l_u.shape, l_uu.shape, l_x.shape, l_xx.shape, l_ux.shape)
		l_x, l_xx = self.clip_derivatives(l_x), self.clip_derivatives(l_xx)
		l_u, l_uu = self.clip_derivatives(l_u), self.clip_derivatives(l_uu)
		l_ux = self.clip_derivatives(l_ux)

		return l_x, l_xx, l_u, l_uu, l_ux
	
	def get_cost_derivatives_short(self, state, control, ref_traj, npc_traj, t):
		"""
		Returns the different cost terms for the trajectory
		This is the main function which calls all the other functions 
		"""
		l_u, l_uu = self.get_control_cost_derivatives(state, control, self.args.horizon_short)
		# l_u, l_uu = np.zeros((4)), np.zeros((4, 4))
		l_x, l_xx = self.get_state_cost_derivatives_short(state, ref_traj, npc_traj, t)
		l_ux = np.zeros((self.args.num_ctrls, self.args.num_states, self.args.horizon_short))
		# print(l_u.shape, l_uu.shape, l_x.shape, l_xx.shape, l_ux.shape)
		l_x, l_xx = self.clip_derivatives(l_x), self.clip_derivatives(l_xx)
		l_u, l_uu = self.clip_derivatives(l_u), self.clip_derivatives(l_uu)
		l_ux = self.clip_derivatives(l_ux)
		
		return l_x, l_xx, l_u, l_uu, l_ux
	
	def compute_obstacle_cost_derivatives(self, npc_traj, i, agent_state):
		r = self.r*2
		P1 = np.diag([1/r**2, 1/r**2, 1/r**2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
		diff = self.clip_derivatives((agent_state - npc_traj[:, i]).reshape(-1, 1))
		# print(diff)
		# print(diff)
		# print(agent_state.shape, npc_traj.shape)
		c = 1 - diff.T @ P1 @ diff
		c_dot = -2 * P1 @ diff
		
		_, b_dot_obs, b_ddot_obs = self.barrier_function(self.args.q1_rear, self.args.q2_rear, c, c_dot)

		return b_dot_obs, b_ddot_obs

	def get_obstacle_cost_derivatives_short(self, state, npc_traj):
		H = self.args.horizon
		l_x = np.zeros((self.args.num_states, H))
		l_xx = np.zeros((self.args.num_states, self.args.num_states, H))
		for i in range(H):
			l_x_i = 0
			# Compute second order derivative
			l_xx_i = 0
			# Obstacle derivative
			l_b_dot_obs = []
			l_b_ddot_obs = []
			npc_traj_k = np.squeeze(npc_traj, axis=0) if len(npc_traj.shape) == 3 else npc_traj
			b_dot_obs_k, b_ddot_obs_k = self.compute_obstacle_cost_derivatives(npc_traj_k, i, state[:, i])

			l_b_dot_obs.append(b_dot_obs_k)
			l_b_ddot_obs.append(b_ddot_obs_k)
			b_dot_obs = np.sum(l_b_dot_obs, axis=0)
			b_ddot_obs = np.sum(l_b_ddot_obs, axis=0)
   
			l_x_i += b_dot_obs.squeeze()
			l_xx_i += b_ddot_obs
    
			l_xx[:, :, i] = l_xx_i
			l_x[:, i] = l_x_i
		return l_x, l_xx


	def get_total_cost(self, state, control_seq, poly_coeffs_y, poly_coeffs_z, x_local_plan):
		"""
		Returns cost of a sequence
		"""
		J = 0
		for i in range(self.args.horizon):
			x_r, y_r, z_r = self.find_closest_point_3d(state[:, i], poly_coeffs_y, poly_coeffs_z, x_local_plan)
			ref_state = np.array([x_r, y_r, z_r, 0, 0, 0, 0, 0, 0, 0, 0, 0]) 
			state_diff = state[:,i]-ref_state

			c_state = state_diff.T @ self.state_cost_c @ state_diff
			c_ctrl = control_seq[:,i].T @ self.control_cost_c @ control_seq[:,i]
			# print(c_state, c_ctrl, end='\r')
			J = J + c_state + c_ctrl
		return J
	
	def get_total_cost_test(self, state, control_seq, poly_coeffs_y, poly_coeffs_z, x_local_plan, id):
		"""
		Returns cost of a sequence
		"""
		J = 0
		for i in range(self.args.horizon):
			x_r, y_r, z_r = self.find_closest_point_3d(state[:, i], poly_coeffs_y, poly_coeffs_z, x_local_plan)
			ref_state = np.array([x_r, y_r, z_r, 0, 0, 0, 0, 0, 0, 0, 0, 0]) 
			state_diff = state[:,i]-ref_state

			c_state = state_diff.T @ self.state_cost_c @ state_diff
			c_ctrl = control_seq[:,i].T @ self.control_cost_c @ control_seq[:,i]

			J = J + c_state + c_ctrl
		return J
	
	def get_total_cost_short(self, state, control_seq, ref_traj, t):
		"""
		Returns cost of a sequence
		"""
		J = 0
		for i in range(self.args.horizon_short):
			x_r, y_r, z_r = ref_traj[0, i], ref_traj[1, i], ref_traj[2, i]
			ref_state = np.array([x_r, y_r, z_r, 0, 0, 0, 0, 0, 0, 0, 0, 0]) 
			state_diff = state[:,i]-ref_state

			c_state = state_diff.T @ self.state_cost_t[i+t] @ state_diff
			c_ctrl = control_seq[:,i].T @ self.control_cost_c @ control_seq[:,i]

			J = J + c_state + c_ctrl
		return J

	def clip_derivatives(self, deri, clip_max=1e7):
		return np.nan_to_num(np.clip(deri, -clip_max, clip_max), copy=False)

	def find_closest_point(self, state, coeffs, x_local_plan):
		new_x = np.linspace(x_local_plan[0], x_local_plan[-1], num=10*self.args.number_of_local_wpts)
		new_y = np.polyval(np.poly1d(coeffs), new_x)
		local_plan = np.vstack((new_x, new_y)).T

		closest_ind = np.sum((local_plan - [state[0], state[1]])**2, axis=1)
		min_i = np.argmin(closest_ind)
		
		return local_plan[min_i, :]
	
	def find_closest_point_3d(self, state, coeffs_y, coeffs_z, x_local_plan):
		# 插值生成更加密集的路径点
		new_x = np.linspace(x_local_plan[0], x_local_plan[-1], num=10*self.args.number_of_local_wpts)
		new_y = np.polyval(np.poly1d(coeffs_y), new_x)
		new_z = np.polyval(np.poly1d(coeffs_z), new_x)
		local_plan = np.vstack((new_x, new_y, new_z)).T
		# print(local_plan.shape)
		# 计算无人机当前位置与路径点之间的欧氏距离平方
		closest_ind = np.sum((local_plan - np.array([state[0], state[1], state[2]]))**2, axis=1)
		min_i = np.argmin(closest_ind)
		ref_ind = min_i + 20 if min_i + 20 < len(local_plan)-1 else len(local_plan)-1
		# print(ref_ind)
		# 返回距离最近的路径点的坐标
		return local_plan[ref_ind, :]

class SAMVGaussian():
    def __init__(self):
        self.D = None
        self.Sigma = None
        self.mu = None
        
    def get_marginal_distrib(self,dim, dim_out=None, reg=1e-2):
        # print(dim, dim_out)
        if dim_out is not None:
            mean_, covariance_ = (self.mu[dim],self.Sigma[dim,dim_out])
        else:
            mean_, covariance_ = (self.mu[dim],self.Sigma[dim,dim] + reg*np.eye(self.Sigma[dim,dim].shape[0]))
        
        return mean_,covariance_
	
    def get_condition_distrib(self, x0, dim1 ,dim2):
        mu1, Sigma11 = self.get_marginal_distrib(dim1)
        mu2, Sigma22 = self.get_marginal_distrib(dim2)
        _, Sigma12 = self.get_marginal_distrib(dim = dim1, dim_out = dim2)
        # print("mu_out:", mu_out.shape, "sigma_in_out:", sigma_in_out.shape, "sigma_in:", sigma_in.shape)
        # print("x_in:", x_in.shape, "mu_in:", mu_in.shape)
        mu_cond = mu2 + Sigma12.T @ np.linalg.inv(Sigma11) @ ((x0-mu1).T)
        mu_cond = mu_cond.flatten()
        Sigma_cond = Sigma22 - np.dot(Sigma12.T, np.dot(np.linalg.inv(Sigma11), Sigma12))
        return mu_cond, Sigma_cond
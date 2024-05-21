import numpy as np 

class Constraints:
	def __init__(self, args, obstacle_bb):
		self.args = args
		self.control_cost = np.array([[self.args.w_acc,                   0],
									  [              0, self.args.w_yawrate]])

		self.state_cost = np.array([[self.args.w_pos, 0, 0, 0],
									[0, self.args.w_pos, 0, 0],
									[0, 0, self.args.w_vel, 0],
									[0, 0, 0,               0]])
		self.coeffs = None

		# self.number_of_npc = 1 # hardcode
   
		self.car_length = obstacle_bb[0]
		self.car_width = obstacle_bb[1]

	def get_state_cost_derivatives(self, state, poly_coeffs, x_local_plan, npc_traj):
		"""
		Returns the first order and second order derivative of the value function wrt state
		"""
		l_x = np.zeros((self.args.num_states, self.args.horizon))
		l_xx = np.zeros((self.args.num_states, self.args.num_states, self.args.horizon))
		for i in range(self.args.horizon):
			# Offset in path derivative
			x_r, y_r = self.find_closest_point(state[:, i], poly_coeffs, x_local_plan)
			traj_cost = 2*self.state_cost@(np.array([state[0, i]-x_r, state[1, i]-y_r, state[2, i]-self.args.desired_speed, 0]))
			# Compute first order derivative
			l_x_i = traj_cost
			# Compute second order derivative
			l_xx_i = 2*self.state_cost
			# Obstacle derivative
			l_b_dot_obs = []
			l_b_ddot_obs = []
			for k in range(len(npc_traj)):
				# print(len(npc_traj))
				npc_traj_k = np.squeeze(npc_traj[k], axis=0) if len(npc_traj[k].shape) == 3 else npc_traj[k]
    
				b_dot_obs_k, b_ddot_obs_k = self.get_obstacle_cost_derivatives(npc_traj_k, i, state[:, i])
				l_b_dot_obs.append(b_dot_obs_k)
				l_b_ddot_obs.append(b_ddot_obs_k)
    
			b_dot_obs = np.sum(l_b_dot_obs, axis=0)
			b_ddot_obs = np.sum(l_b_ddot_obs, axis=0)
   
			l_x_i += b_dot_obs.squeeze()
			l_xx_i += b_ddot_obs
    
			l_xx[:, :, i] = l_xx_i
			l_x[:, i] = l_x_i
		return l_x, l_xx

	def get_control_cost_derivatives(self, state, control):
		"""
		Returns the control quadratic (R matrix) and linear cost term (r vector) for the trajectory
		"""
		P1 = np.array([[1],[0]])
		P2 = np.array([[0],[1]])

		l_u = np.zeros((self.args.num_ctrls, self.args.horizon))
		l_uu = np.zeros((self.args.num_ctrls, self.args.num_ctrls, self.args.horizon))
		# c_ctrl = 0
		for i in range(self.args.horizon):
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

			l_u_i = b_dot_1 + b_dot_2 + b_dot_3 + b_dot_4 + (2*control[:, i].T @ self.control_cost).reshape(-1, 1)
			l_uu_i = b_ddot_1 + b_ddot_2 + b_ddot_3 + b_ddot_4 + 2*self.control_cost

			l_u[:, i] = l_u_i.squeeze()
			l_uu[:, :, i] = l_uu_i.squeeze()

		return l_u, l_uu
    
	def barrier_function(self, q1, q2, c, c_dot):
		b = q1*np.exp(q2*c)
		b_dot = q1*q2*np.exp(q2*c)*c_dot
		b_ddot = q1*(q2**2)*np.exp(q2*c)*np.matmul(c_dot, c_dot.T)

		return b, b_dot, b_ddot

	def get_cost_derivatives(self, state, control, poly_coeffs, x_local_plan, npc_traj):
		"""
		Returns the different cost terms for the trajectory
		This is the main function which calls all the other functions 
		"""
		l_u, l_uu = self.get_control_cost_derivatives(state, control)
		l_x, l_xx = self.get_state_cost_derivatives(state, poly_coeffs, x_local_plan, npc_traj)
		l_ux = np.zeros((self.args.num_ctrls, self.args.num_states, self.args.horizon))
		# print(l_u.shape, l_uu.shape, l_x.shape, l_xx.shape, l_ux.shape)
		return l_x, l_xx, l_u, l_uu, l_ux

	def get_obstacle_cost_derivatives(self, npc_traj, i, agent_state):
		a = self.car_length + np.abs(npc_traj[2, i]*np.cos(npc_traj[3, i]))*self.args.t_safe + self.args.s_safe_a + self.args.agent_rad
		b = self.car_width + np.abs(npc_traj[2, i]*np.sin(npc_traj[3, i]))*self.args.t_safe + self.args.s_safe_b + self.args.agent_rad
        
		P1 = np.diag([1/a**2, 1/b**2, 0, 0])

		theta = npc_traj[3, i]
		theta_agent = agent_state[3]

		transformation_matrix = np.array([[ np.cos(theta), np.sin(theta), 0, 0],
                                          [-np.sin(theta), np.cos(theta), 0, 0],
                                          [               0,               0, 0, 0],
                                          [               0,               0, 0, 0]])
        
		agent_front = agent_state + np.array([np.cos(theta_agent)*self.args.agent_lf, np.sin(theta_agent)*self.args.agent_lf, 0, 0])
		diff = (transformation_matrix @ (agent_front - npc_traj[:, i])).reshape(-1, 1) # (x- xo)
		c = 1 - diff.T @ P1 @ diff # Transform into a constraint function
		c_dot = -2 * P1 @ diff
		_, b_dot_f, b_ddot_f = self.barrier_function(self.args.q1_front, self.args.q2_front, c, c_dot)

		agent_rear = agent_state - np.array([np.cos(theta_agent)*self.args.agent_lr, np.sin(theta_agent)*self.args.agent_lr, 0, 0])
		diff = (transformation_matrix @ (agent_rear - npc_traj[:, i])).reshape(-1, 1)
		c = 1 - diff.T @ P1 @ diff
		c_dot = -2 * P1 @ diff
		_, b_dot_r, b_ddot_r = self.barrier_function(self.args.q1_rear, self.args.q2_rear, c, c_dot)

		b_dot_obs = b_dot_f + b_dot_r
		b_ddot_obs = b_ddot_f + b_ddot_r
		return b_dot_obs, b_ddot_obs

	def get_total_cost(self, state, control_seq, poly_coeffs, x_local_plan):
		"""
		Returns cost of a sequence
		"""
		J = 0
		for i in range(self.args.horizon):
			x_r, y_r = self.find_closest_point(state[:, i], poly_coeffs, x_local_plan)
			ref_state = np.array([x_r, y_r, self.args.desired_speed, 0]) # Theta does not matter
			state_diff = state[:,i]-ref_state

			c_state = state_diff.T @ self.state_cost @ state_diff
			c_ctrl = control_seq[:,i].T @ self.control_cost @ control_seq[:,i]

			J = J + c_state + c_ctrl
		return J

	def find_closest_point(self, state, coeffs, x_local_plan):
		new_x = np.linspace(x_local_plan[0], x_local_plan[-1], num=10*self.args.number_of_local_wpts)
		new_y = np.polyval(np.poly1d(coeffs), new_x)
		local_plan = np.vstack((new_x, new_y)).T

		closest_ind = np.sum((local_plan - [state[0], state[1]])**2, axis=1)
		min_i = np.argmin(closest_ind)
		
		return local_plan[min_i, :]
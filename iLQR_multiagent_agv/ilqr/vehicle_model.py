import numpy as np
import ctypes

# Add lambda functions
cos = lambda a : np.cos(a)
sin = lambda a : np.sin(a)
tan = lambda a : np.tan(a)

class Model:
    """
    A vehicle model with 4 dof. 
    State - [x, y, velocity, theta]
    Control - [acc, yaw_rate]
    """
    def __init__(self, args):
        self.wheelbase = args.wheelbase
        self.steer_min = args.steer_angle_limits[0] # 转向角限制
        self.steer_max = args.steer_angle_limits[1] 
        self.accel_min = args.acc_limits[0] # 加速度限制
        self.accel_max = args.acc_limits[1]
        self.max_speed = args.max_speed # 最大速度
        self.Ts = args.timestep 
        self.N = args.horizon # T=horizon*timesteps
        self.zeros = np.zeros((self.N)) 
        self.ones = np.ones((self.N))
        self.lib = ctypes.CDLL('./bicycle_dynamics/compute_dynamics.so')
        self.lib.compute_A_matrix.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # A (output)
            ctypes.POINTER(ctypes.c_double),  # v
            ctypes.POINTER(ctypes.c_double),  # theta
            ctypes.POINTER(ctypes.c_double),  # v_dot(a)
            ctypes.c_int,                     # horizon
            ctypes.c_double                   # dt
        ]
        self.lib.compute_B_matrix.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # A (output)
            ctypes.POINTER(ctypes.c_double),  # theta
            ctypes.c_int,                     # horizon
            ctypes.c_double                   # dt
        ]
        self.lib.compute_next_state.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # x_next
            ctypes.POINTER(ctypes.c_double),  # px
            ctypes.POINTER(ctypes.c_double),  # py
            ctypes.POINTER(ctypes.c_double),  # v
            ctypes.POINTER(ctypes.c_double),  # theta
            ctypes.POINTER(ctypes.c_double),  # v_dot(a)
            ctypes.POINTER(ctypes.c_double),  # theta_dot
            ctypes.c_int,                     # horizon
            ctypes.c_double                   # dt
        ]

        self.lib.compute_A_matrix.restype = None
        self.lib.compute_B_matrix.restype = None
        self.lib.compute_next_state.restype = None

    def forward_simulate(self, state, control):
        """
        Find the next state of the vehicle given the current state and control input
        """
        # Clips the controller values between min and max accel and steer values
        control[0] = np.clip(control[0], self.accel_min, self.accel_max)
        control[1] = np.clip(control[1], state[2]*tan(self.steer_min)/self.wheelbase, state[2]*tan(self.steer_max)/self.wheelbase)
        
        next_state = np.array([state[0] + cos(state[3])*(state[2]*self.Ts + (control[0]*self.Ts**2)/2),
                               state[1] + sin(state[3])*(state[2]*self.Ts + (control[0]*self.Ts**2)/2),
                               np.clip(state[2] + control[0]*self.Ts, 0.0, self.max_speed),
                               state[3] + control[1]*self.Ts])  # wrap angles between 0 and 2*pi
        return next_state
    
    # def forward_simulate(self, x, u):
    #     """
    #     Find the next state of the vehicle given the current state and control input
    #     """
    #     # Clips the controller values between min and max accel and steer values
    #     # control[0] = np.clip(control[0], self.accel_min, self.accel_max)
    #     # control[1] = np.clip(control[1], state[2]*tan(self.steer_min)/self.wheelbase, state[2]*tan(self.steer_max)/self.wheelbase)
    #     x = x.reshape(4, -1); u = u.reshape(2, -1)
    #     next_state = np.zeros(4)
    #     px, py, v, theta = x[0], x[1], x[2], x[3]
    #     a, theta_dot = u[0], u[1]
    #     self.lib.compute_next_state(
    #         next_state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    #         px.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    #         py.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    #         v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    #         theta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    #         a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    #         theta_dot.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    #         1, self.Ts
    #     )
    #     return next_state.reshape(-1)

    # def get_A_matrix(self, velocity_vals, theta, acceleration_vals, horizon):
    #     """
    #     Returns the linearized 'A' matrix of the agent vehicle 
    #     model for all states in backward pass. 
    #     """
    #     zeros = np.zeros((horizon)) 
    #     ones = np.ones((horizon))
    #     v = velocity_vals
    #     v_dot = acceleration_vals
    #     A = np.array([[ones, zeros, cos(theta)*self.Ts, -(v*self.Ts + (v_dot*self.Ts**2)/2)*sin(theta)],
    #                   [zeros, ones, sin(theta)*self.Ts,  (v*self.Ts + (v_dot*self.Ts**2)/2)*cos(theta)],
    #                   [zeros, zeros,             ones,                                         zeros],
    #                   [zeros, zeros,             zeros,                                        ones]])
    #     return A
    
    def get_A_matrix(self, velocity_vals, theta, acceleration_vals, horizon):
        """
        Returns the linearized 'A' matrix of the agent vehicle 
        model for all states in backward pass. 
        """
        A = np.zeros((4, 4, horizon))
        self.lib.compute_A_matrix(
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            velocity_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            theta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            acceleration_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            horizon, self.Ts
        )
        return A
    
    # def get_B_matrix(self, theta, horizon):
    #     """
    #     Returns the linearized 'B' matrix of the agent vehicle 
    #     model for all states in backward pass. 
    #     """
    #     zeros = np.zeros((horizon)) 
    #     ones = np.ones((horizon))

    #     B = np.array([[self.Ts**2*cos(theta)/2,        zeros],
    #                   [self.Ts**2*sin(theta)/2,        zeros],
    #                   [         self.Ts*ones,         zeros],
    #                   [                 zeros, self.Ts*ones]])
    #     return B

    def get_B_matrix(self, theta, horizon):
        """
        Returns the linearized 'B' matrix of the agent vehicle 
        model for all states in backward pass. 
        """
        B = np.zeros((4, 2, horizon))
        self.lib.compute_B_matrix(
            B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            theta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            horizon, self.Ts
        )
        return B
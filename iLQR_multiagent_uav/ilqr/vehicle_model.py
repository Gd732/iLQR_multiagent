import numpy as np
import ctypes

# Add lambda functions
cos = lambda a : np.cos(a)
sin = lambda a : np.sin(a)
tan = lambda a : np.tan(a)

class BicycleModel:
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

    def get_A_matrix(self, velocity_vals, theta, acceleration_vals, horizon):
        """
        Returns the linearized 'A' matrix of the agent vehicle 
        model for all states in backward pass. 
        """
        zeros = np.zeros((horizon)) 
        ones = np.ones((horizon))
        v = velocity_vals
        v_dot = acceleration_vals
        A = np.array([[ones, zeros, cos(theta)*self.Ts, -(v*self.Ts + (v_dot*self.Ts**2)/2)*sin(theta)],
                      [zeros, ones, sin(theta)*self.Ts,  (v*self.Ts + (v_dot*self.Ts**2)/2)*cos(theta)],
                      [zeros, zeros,             ones,                                         zeros],
                      [zeros, zeros,             zeros,                                        ones]])
        return A

    def get_B_matrix(self, theta, horizon):
        """
        Returns the linearized 'B' matrix of the agent vehicle 
        model for all states in backward pass. 
        """
        zeros = np.zeros((horizon)) 
        ones = np.ones((horizon))

        B = np.array([[self.Ts**2*cos(theta)/2,        zeros],
                      [self.Ts**2*sin(theta)/2,        zeros],
                      [         self.Ts*ones,         zeros],
                      [                 zeros, self.Ts*ones]])
        return B


class QuadcopterModel:
    def __init__(self, args, I = np.diag(np.array([2,2,4])), kd = 1, 
                 k = 1, L = 0.3, b = 1, m=1, g=9.81, Dx=12, Du=4):
        self.I = I #inertia
        self.kd = kd #friction
        self.k = k #motor constant
        self.L = L# distance between center and motor
        self.b = b # drag coefficient
        self.m = m # mass
        self.g = g
        self.Dx = Dx
        self.Du = Du
        self.dt = args.timestep

        self.lib = ctypes.CDLL('./drone_dynamics/compute_dynamics.so')
        self.lib.compute_B_matrix.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # B (output)
            ctypes.POINTER(ctypes.c_double),  # tx
            ctypes.POINTER(ctypes.c_double),  # ty
            ctypes.POINTER(ctypes.c_double),  # tz
            ctypes.c_int,                     # horizon
            ctypes.c_double,                  # Ix
            ctypes.c_double,                  # Iy
            ctypes.c_double,                  # Iz
            ctypes.c_double,                  # m
            ctypes.c_double                   # dt
        ]
        self.lib.compute_A_matrix.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # A (output)
            ctypes.POINTER(ctypes.c_double),  # tx
            ctypes.POINTER(ctypes.c_double),  # ty
            ctypes.POINTER(ctypes.c_double),  # tz
            ctypes.POINTER(ctypes.c_double),  # tdotx
            ctypes.POINTER(ctypes.c_double),  # tdoty
            ctypes.POINTER(ctypes.c_double),  # tdotz
            ctypes.POINTER(ctypes.c_double),  # Tau
            ctypes.POINTER(ctypes.c_double),  # taux
            ctypes.POINTER(ctypes.c_double),  # tauy
            ctypes.POINTER(ctypes.c_double),  # tauz
            ctypes.c_int,                     # horizon
            ctypes.c_double,                  # Ix
            ctypes.c_double,                  # Iy
            ctypes.c_double,                  # Iz
            ctypes.c_double,                  # kd
            ctypes.c_double,                  # m
            ctypes.c_double                   # dt
        ]
        self.lib.compute_next_state.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # x_next
            ctypes.POINTER(ctypes.c_double),  # px
            ctypes.POINTER(ctypes.c_double),  # py
            ctypes.POINTER(ctypes.c_double),  # pz
            ctypes.POINTER(ctypes.c_double),  # pdotx
            ctypes.POINTER(ctypes.c_double),  # pdoty
            ctypes.POINTER(ctypes.c_double),  # pdotz
            ctypes.POINTER(ctypes.c_double),  # tx
            ctypes.POINTER(ctypes.c_double),  # ty
            ctypes.POINTER(ctypes.c_double),  # tz
            ctypes.POINTER(ctypes.c_double),  # tdotx
            ctypes.POINTER(ctypes.c_double),  # tdoty
            ctypes.POINTER(ctypes.c_double),  # tdotz
            ctypes.POINTER(ctypes.c_double),  # Tau
            ctypes.POINTER(ctypes.c_double),  # taux
            ctypes.POINTER(ctypes.c_double),  # tauy
            ctypes.POINTER(ctypes.c_double),  # tauz
            ctypes.c_int,                     # horizon
            ctypes.c_double,                  # Ix
            ctypes.c_double,                  # Iy
            ctypes.c_double,                  # Iz
            ctypes.c_double,                  # kd
            ctypes.c_double,                  # m
            ctypes.c_double                   # dt
        ]

        self.lib.compute_A_matrix.restype = None
        self.lib.compute_B_matrix.restype = None
        self.lib.compute_next_state.restype = None

    def thrust(self, inputs):
        T = np.array([0,0, self.k*np.sum(inputs)])
        return T

    def torques(self, inputs):
        tau = np.array([self.L*self.k*(inputs[0]-inputs[2]), self.L*self.k*(inputs[1]-inputs[3]), \
                        self.b*(inputs[0]-inputs[1] + inputs[2] - inputs[3])])
        return tau

    def acceleration(self, inputs, angles, xdot):
        gravity = np.array([0,0,-self.g])
        R = self.Rotation(angles)
        T = R.dot(self.thrust(inputs))
        Fd = -self.kd*xdot
        a = gravity + T/self.m + Fd
        return a

    def angular_acceleration(self, inputs, omega):
        tau = self.torques(inputs)
        omegadot = np.linalg.inv(self.I).dot(tau - np.cross(omega, self.I.dot(omega)))
        return omegadot

    def thetadot2omega(self, thetadot, theta):
        s0, c0 = np.sin(theta[0]), np.cos(theta[0])
        s1, c1 = np.sin(theta[1]), np.cos(theta[1])        
        R = np.array([[1, 0, -s1], \
                     [0, c0, c1*s0], \
                     [0, -s0, c1*c0]])
        return R @ thetadot

    def omega2thetadot(self, omega, theta):
        s0, c0 = np.sin(theta[0]), np.cos(theta[0])
        s1, c1 = np.sin(theta[1]), np.cos(theta[1])
        # R = np.array([[1, 0, -np.sin(theta[1])], \
        #              [0, np.cos(theta[0]), np.cos(theta[1])*np.sin(theta[0])], \
        #              [0, -np.sin(theta[0]), np.cos(theta[1])*np.cos(theta[0])]])
        R_inv = np.array([[1, s0*s1/c1, c0*s1/c1], \
                            [0, c0, -s0], \
                            [0, s0/c1, c0/c1]])
        return R_inv @ omega

    def Rotation(self, theta):
        c0,s0 = np.cos(theta[0]), np.sin(theta[0])
        c1,s1 = np.cos(theta[1]), np.sin(theta[1])
        c2,s2 = np.cos(theta[2]), np.sin(theta[2])

        R = np.array([[c0*c2 - c1*s0*s2, -c2*s0 - c0*c1*s2, s1 * s2], 
                     [c1*c2*s0 + c0*s2, c0*c1*c2-s0*s2, -c2*s1], 
                     [s0*s1, c0*s1, c1]])
        return R
        
    def set_init_state(self,x0):
        self.x0 = x0

    def compute_matrices(self,x,u, inc = 0.001):
        Dx, Du = len(x), len(u)
        A = np.zeros((Dx, Dx))
        B = np.zeros((Dx, Du))
        
        xnext = self.forward_simulate(x, u)
        for i in range(Dx):
            xp, xm = x.copy(), x.copy()
            xp[i] += inc
            xnextp = self.forward_simulate(xp, u)
            xm[i] -= inc
            xnextm = self.forward_simulate(xm, u)
            diff = (xnextp - xnextm)/(2*inc)
            A[:,i] = diff
            
        for i in range(Du):
            up, um = u.copy(), u.copy()
            up[i] += inc
            xnextp = self.forward_simulate(x, up)
            um[i] -= inc
            xnextm = self.forward_simulate(x, um)
            diff = (xnextp - xnextm)/(2*inc)
            B[:,i] = diff
        
        return A,B
    
    # def get_A_matrix(self, x, u, horizon, inc = 0.01):
    #     A = np.zeros((self.Dx, self.Dx, horizon))
    #     for t in range(horizon):
    #         for i in range(self.Dx):
    #             xp, xm = x[:,t].copy(), x[:,t].copy()
    #             xp[i] += inc
    #             xnextp = self.forward_simulate(xp, u[:,t])
    #             xm[i] -= inc
    #             xnextm = self.forward_simulate(xm, u[:,t])
    #             diff = (xnextp - xnextm)/(2*inc)
    #             A[:,i,t] = diff
    #     return A
 
    def get_A_matrix_py(self, x, u, horizon):
        A = np.zeros((self.Dx, self.Dx, horizon))
        Ix = self.I[0, 0]; Iy = self.I[1, 1]; Iz = self.I[2, 2]
        m = self.m
        dt = self.dt; kd = self.kd
        tx, ty, tz = x[6], x[7], x[8]
        tdotx, tdoty, tdotz = x[9], x[10], x[11]
        Tau, taux, tauy, tauz = u[0], u[1], u[2], u[3]

        A[0, 0, :] = 1
        A[0, 3, :] = dt
        A[1, 1, :] = 1
        A[1, 4, :] = dt
        A[2, 2, :] = 1
        A[2, 5, :] = dt
        A[3, 3, :] = 1 - dt*kd
        A[3, 7, :] = (Tau*dt*cos(ty)*sin(tz))/m
        A[3, 8, :] = (Tau*dt*cos(tz)*sin(ty))/m
        A[4, 4, :] = 1 - dt*kd
        A[4, 7, :] = -(Tau*dt*cos(ty)*cos(tz))/m
        A[4, 8, :] = (Tau*dt*sin(ty)*sin(tz))/m
        A[5, 5, :] = 1 - dt*kd
        A[5, 7, :] = -(Tau*dt*sin(ty))/m
        A[6, 6, :] = (Ix*Iy*Iz*cos(ty) - Iy*Iz**2*dt**2*tdoty**2*cos(ty)*sin(tx)**2 + Iy**2*Iz*dt**2*tdoty**2*cos(ty)*sin(tx)**2 - Iy*Iz**2*dt**2*tdotz**2*cos(tx)**2*cos(ty)**3 + Iy**2*Iz*dt**2*tdotz**2*cos(tx)**2*cos(ty)**3 + Iy*Iz**2*dt**2*tdotz**2*cos(ty)**3*sin(tx)**2 - Iy**2*Iz*dt**2*tdotz**2*cos(ty)**3*sin(tx)**2 + Ix*Iz*dt**2*tauy*cos(tx)*sin(ty) + Iy*Iz**2*dt**2*tdoty**2*cos(tx)**2*cos(ty) - Iy**2*Iz*dt**2*tdoty**2*cos(tx)**2*cos(ty) - Ix*Iy*dt**2*tauz*sin(tx)*sin(ty) + Ix*Iy**2*dt**2*tdotz**2*cos(tx)**2*cos(ty)*sin(ty)**2 - Ix**2*Iy*dt**2*tdotz**2*cos(tx)**2*cos(ty)*sin(ty)**2 - Ix*Iz**2*dt**2*tdotz**2*cos(tx)**2*cos(ty)*sin(ty)**2 + Ix**2*Iz*dt**2*tdotz**2*cos(tx)**2*cos(ty)*sin(ty)**2 - Ix*Iy**2*dt**2*tdotz**2*cos(ty)*sin(tx)**2*sin(ty)**2 + Ix**2*Iy*dt**2*tdotz**2*cos(ty)*sin(tx)**2*sin(ty)**2 + Ix*Iz**2*dt**2*tdotz**2*cos(ty)*sin(tx)**2*sin(ty)**2 - Ix**2*Iz*dt**2*tdotz**2*cos(ty)*sin(tx)**2*sin(ty)**2 - 2*Ix*Iy**2*dt**2*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty)**2 + 2*Ix**2*Iy*dt**2*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty)**2 + Ix*Iy**2*dt**2*tdotx*tdotz*cos(ty)*sin(tx)**2*sin(ty) - Ix**2*Iy*dt**2*tdotx*tdotz*cos(ty)*sin(tx)**2*sin(ty) + 2*Ix*Iz**2*dt**2*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty)**2 - 2*Ix**2*Iz*dt**2*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty)**2 - Ix*Iz**2*dt**2*tdotx*tdotz*cos(ty)*sin(tx)**2*sin(ty) + Ix**2*Iz*dt**2*tdotx*tdotz*cos(ty)*sin(tx)**2*sin(ty) + 2*Ix*Iy**2*dt**2*tdotx*tdoty*cos(tx)*sin(tx)*sin(ty) - 2*Ix**2*Iy*dt**2*tdotx*tdoty*cos(tx)*sin(tx)*sin(ty) - 2*Ix*Iz**2*dt**2*tdotx*tdoty*cos(tx)*sin(tx)*sin(ty) + 2*Ix**2*Iz*dt**2*tdotx*tdoty*cos(tx)*sin(tx)*sin(ty) - Ix*Iy**2*dt**2*tdotx*tdotz*cos(tx)**2*cos(ty)*sin(ty) + Ix**2*Iy*dt**2*tdotx*tdotz*cos(tx)**2*cos(ty)*sin(ty) + Ix*Iz**2*dt**2*tdotx*tdotz*cos(tx)**2*cos(ty)*sin(ty) - Ix**2*Iz*dt**2*tdotx*tdotz*cos(tx)**2*cos(ty)*sin(ty) + 4*Iy*Iz**2*dt**2*tdoty*tdotz*cos(tx)*cos(ty)**2*sin(tx) - 4*Iy**2*Iz*dt**2*tdoty*tdotz*cos(tx)*cos(ty)**2*sin(tx))/(Ix*Iy*Iz*cos(ty))
        A[6, 7, :] = (dt*(Ix**2*Iz*dt*tdotx*tdoty - Ix*Iz**2*dt*tdotx*tdoty + Ix*Iy*dt*tauz*cos(tx) + Ix*Iz*dt*tauy*sin(tx) + Ix*Iz**2*dt*tdoty*tdotz*sin(ty) - Ix**2*Iz*dt*tdoty*tdotz*sin(ty) - Ix*Iy**2*dt*tdotx*tdoty*cos(tx)**2 + Ix**2*Iy*dt*tdotx*tdoty*cos(tx)**2 + Ix*Iz**2*dt*tdotx*tdoty*cos(tx)**2 - Ix**2*Iz*dt*tdotx*tdoty*cos(tx)**2 + Ix*Iy**2*dt*tdoty*tdotz*cos(tx)**2*sin(ty) - Ix**2*Iy*dt*tdoty*tdotz*cos(tx)**2*sin(ty) - Ix*Iz**2*dt*tdoty*tdotz*cos(tx)**2*sin(ty) + Ix**2*Iz*dt*tdoty*tdotz*cos(tx)**2*sin(ty) + Ix*Iz**2*dt*tdoty*tdotz*cos(ty)**2*sin(ty) - Ix**2*Iz*dt*tdoty*tdotz*cos(ty)**2*sin(ty) - Iy*Iz**2*dt*tdoty*tdotz*cos(ty)**2*sin(ty) + Iy**2*Iz*dt*tdoty*tdotz*cos(ty)**2*sin(ty) + 2*Ix*Iy**2*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) - 2*Ix**2*Iy*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) - 2*Ix*Iz**2*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) + 2*Ix**2*Iz*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) + 2*Iy*Iz**2*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) - 2*Iy**2*Iz*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) - Ix*Iy**2*dt*tdotx*tdotz*cos(tx)*cos(ty)**3*sin(tx) + Ix**2*Iy*dt*tdotx*tdotz*cos(tx)*cos(ty)**3*sin(tx) + Ix*Iz**2*dt*tdotx*tdotz*cos(tx)*cos(ty)**3*sin(tx) - Ix**2*Iz*dt*tdotx*tdotz*cos(tx)*cos(ty)**3*sin(tx) + Ix*Iy**2*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty) - Ix**2*Iy*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty) - Ix*Iz**2*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty) + Ix**2*Iz*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty) + 2*Iy*Iz**2*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty) - 2*Iy**2*Iz*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty)))/(Ix*Iy*Iz*cos(ty)**2)
        A[6, 9, :] = (dt*(Iy*Iz*cos(ty) - Iy**2*dt*tdoty*cos(tx)**2*sin(ty) - Iz**2*dt*tdoty*sin(tx)**2*sin(ty) + Ix*Iy*dt*tdoty*cos(tx)**2*sin(ty) + Ix*Iz*dt*tdoty*sin(tx)**2*sin(ty) - Iy**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) + Iz**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) + Ix*Iy*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) - Ix*Iz*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty)))/(Iy*Iz*cos(ty))
        A[6, 10, :] = -(dt**2*(Iy*Iz**2*tdotz*cos(tx)**2*cos(ty)**2 - Iy**2*Iz*tdotz*cos(tx)**2*cos(ty)**2 - Ix*Iy**2*tdotz*cos(tx)**2*sin(ty)**2 + Ix**2*Iy*tdotz*cos(tx)**2*sin(ty)**2 - Iy*Iz**2*tdotz*cos(ty)**2*sin(tx)**2 + Iy**2*Iz*tdotz*cos(ty)**2*sin(tx)**2 - Ix*Iz**2*tdotz*sin(tx)**2*sin(ty)**2 + Ix**2*Iz*tdotz*sin(tx)**2*sin(ty)**2 + Ix*Iy**2*tdotx*cos(tx)**2*sin(ty) - Ix**2*Iy*tdotx*cos(tx)**2*sin(ty) + Ix*Iz**2*tdotx*sin(tx)**2*sin(ty) - Ix**2*Iz*tdotx*sin(tx)**2*sin(ty) - 2*Iy*Iz**2*tdoty*cos(tx)*cos(ty)*sin(tx) + 2*Iy**2*Iz*tdoty*cos(tx)*cos(ty)*sin(tx)))/(Ix*Iy*Iz*cos(ty))
        A[6, 11, :] = (dt*(Ix*Iy*Iz*cos(tx)**2*cos(ty)*sin(ty) - (Ix*Iy*Iz*sin(2*ty))/2 + Ix*Iy*Iz*cos(ty)*sin(tx)**2*sin(ty) - Iy*Iz**2*dt*tdoty*cos(tx)**2*cos(ty)**2 + Iy**2*Iz*dt*tdoty*cos(tx)**2*cos(ty)**2 + Ix*Iy**2*dt*tdoty*cos(tx)**2*sin(ty)**2 - Ix**2*Iy*dt*tdoty*cos(tx)**2*sin(ty)**2 + Iy*Iz**2*dt*tdoty*cos(ty)**2*sin(tx)**2 - Iy**2*Iz*dt*tdoty*cos(ty)**2*sin(tx)**2 + Ix*Iz**2*dt*tdoty*sin(tx)**2*sin(ty)**2 - Ix**2*Iz*dt*tdoty*sin(tx)**2*sin(ty)**2 - 2*Iy*Iz**2*dt*tdotz*cos(tx)*cos(ty)**3*sin(tx) + 2*Iy**2*Iz*dt*tdotz*cos(tx)*cos(ty)**3*sin(tx) + 2*Ix*Iy**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty)**2 - 2*Ix**2*Iy*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty)**2 - 2*Ix*Iz**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty)**2 + 2*Ix**2*Iz*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty)**2 - Ix*Iy**2*dt*tdotx*cos(tx)*cos(ty)*sin(tx)*sin(ty) + Ix**2*Iy*dt*tdotx*cos(tx)*cos(ty)*sin(tx)*sin(ty) + Ix*Iz**2*dt*tdotx*cos(tx)*cos(ty)*sin(tx)*sin(ty) - Ix**2*Iz*dt*tdotx*cos(tx)*cos(ty)*sin(tx)*sin(ty)))/(Ix*Iy*Iz*cos(ty))
        A[7, 6, :] = -(dt**2*(Iz*tauy*sin(tx) + Iy**2*tdotx*tdoty - Iz**2*tdotx*tdoty + Iy*tauz*cos(tx) - 2*Iy**2*tdotx*tdoty*cos(tx)**2 + 2*Iz**2*tdotx*tdoty*cos(tx)**2 - Iy**2*tdoty*tdotz*sin(ty) + Iz**2*tdoty*tdotz*sin(ty) - Ix*Iy*tdotx*tdoty + Ix*Iz*tdotx*tdoty + 2*Iy**2*tdoty*tdotz*cos(tx)**2*sin(ty) - 2*Iz**2*tdoty*tdotz*cos(tx)**2*sin(ty) + 2*Ix*Iy*tdotx*tdoty*cos(tx)**2 - 2*Ix*Iz*tdotx*tdoty*cos(tx)**2 + Ix*Iy*tdoty*tdotz*sin(ty) - Ix*Iz*tdoty*tdotz*sin(ty) + 2*Iy**2*tdotz**2*cos(tx)*cos(ty)*sin(tx)*sin(ty) - 2*Iz**2*tdotz**2*cos(tx)*cos(ty)*sin(tx)*sin(ty) - 2*Iy**2*tdotx*tdotz*cos(tx)*cos(ty)*sin(tx) + 2*Iz**2*tdotx*tdotz*cos(tx)*cos(ty)*sin(tx) - 2*Ix*Iy*tdoty*tdotz*cos(tx)**2*sin(ty) + 2*Ix*Iz*tdoty*tdotz*cos(tx)**2*sin(ty) - 2*Ix*Iy*tdotz**2*cos(tx)*cos(ty)*sin(tx)*sin(ty) + 2*Ix*Iz*tdotz**2*cos(tx)*cos(ty)*sin(tx)*sin(ty) + 2*Ix*Iy*tdotx*tdotz*cos(tx)*cos(ty)*sin(tx) - 2*Ix*Iz*tdotx*tdotz*cos(tx)*cos(ty)*sin(tx)))/(Iy*Iz)
        A[7, 7, :] = (Iy*Iz - Iz**2*dt**2*tdotz**2*cos(tx)**2*cos(ty)**2 - Iy**2*dt**2*tdotz**2*cos(ty)**2*sin(tx)**2 + Iz**2*dt**2*tdotz**2*cos(tx)**2*sin(ty)**2 + Iy**2*dt**2*tdotz**2*sin(tx)**2*sin(ty)**2 + Ix*Iy*dt**2*tdotz**2*cos(ty)**2*sin(tx)**2 - Ix*Iz*dt**2*tdotz**2*cos(tx)**2*sin(ty)**2 - Ix*Iy*dt**2*tdotz**2*sin(tx)**2*sin(ty)**2 - Iz**2*dt**2*tdotx*tdotz*cos(tx)**2*sin(ty) - Iy**2*dt**2*tdotx*tdotz*sin(tx)**2*sin(ty) + Ix*Iz*dt**2*tdotz**2*cos(tx)**2*cos(ty)**2 - Iy**2*dt**2*tdoty*tdotz*cos(tx)*cos(ty)*sin(tx) + Iz**2*dt**2*tdoty*tdotz*cos(tx)*cos(ty)*sin(tx) + Ix*Iz*dt**2*tdotx*tdotz*cos(tx)**2*sin(ty) + Ix*Iy*dt**2*tdotx*tdotz*sin(tx)**2*sin(ty) + Ix*Iy*dt**2*tdoty*tdotz*cos(tx)*cos(ty)*sin(tx) - Ix*Iz*dt**2*tdoty*tdotz*cos(tx)*cos(ty)*sin(tx))/(Iy*Iz)
        A[7, 9, :] = (dt**2*((Iy**2*tdoty*sin(2*tx))/2 - (Iz**2*tdoty*sin(2*tx))/2 + Iz**2*tdotz*cos(tx)**2*cos(ty) + Iy**2*tdotz*cos(ty)*sin(tx)**2 - (Ix*Iy*tdoty*sin(2*tx))/2 + (Ix*Iz*tdoty*sin(2*tx))/2 - Ix*Iz*tdotz*cos(tx)**2*cos(ty) - Ix*Iy*tdotz*cos(ty)*sin(tx)**2))/(Iy*Iz)
        A[7, 10, :] = (dt*(Iy*Iz + (Iy**2*dt*tdotx*sin(2*tx))/2 - (Iz**2*dt*tdotx*sin(2*tx))/2 - (Ix*Iy*dt*tdotx*sin(2*tx))/2 + (Ix*Iz*dt*tdotx*sin(2*tx))/2 - Iy**2*dt*tdotz*cos(tx)*sin(tx)*sin(ty) + Iz**2*dt*tdotz*cos(tx)*sin(tx)*sin(ty) + Ix*Iy*dt*tdotz*cos(tx)*sin(tx)*sin(ty) - Ix*Iz*dt*tdotz*cos(tx)*sin(tx)*sin(ty)))/(Iy*Iz)
        A[7, 11, :] = (dt**2*(Iz**2*tdotx*cos(tx)**2*cos(ty) + Iy**2*tdotx*cos(ty)*sin(tx)**2 - Iy**2*tdoty*cos(tx)*sin(tx)*sin(ty) + Iz**2*tdoty*cos(tx)*sin(tx)*sin(ty) - 2*Iz**2*tdotz*cos(tx)**2*cos(ty)*sin(ty) - Ix*Iz*tdotx*cos(tx)**2*cos(ty) - 2*Iy**2*tdotz*cos(ty)*sin(tx)**2*sin(ty) - Ix*Iy*tdotx*cos(ty)*sin(tx)**2 + Ix*Iy*tdoty*cos(tx)*sin(tx)*sin(ty) - Ix*Iz*tdoty*cos(tx)*sin(tx)*sin(ty) + 2*Ix*Iz*tdotz*cos(tx)**2*cos(ty)*sin(ty) + 2*Ix*Iy*tdotz*cos(ty)*sin(tx)**2*sin(ty)))/(Iy*Iz)
        A[8, 6, :] = -(dt**2*(Iy*tauz*sin(tx) + (Iy**2*tdotz**2*sin(2*ty))/2 - (Iz**2*tdotz**2*sin(2*ty))/2 - Iz*tauy*cos(tx) - (Ix*Iy*tdotz**2*sin(2*ty))/2 + (Ix*Iz*tdotz**2*sin(2*ty))/2 - Iy**2*tdotx*tdoty*sin(2*tx) + Iz**2*tdotx*tdoty*sin(2*tx) - Iy**2*tdotx*tdotz*cos(ty) + Iz**2*tdotx*tdotz*cos(ty) + 2*Iy**2*tdotx*tdotz*cos(tx)**2*cos(ty) - 2*Iz**2*tdotx*tdotz*cos(tx)**2*cos(ty) + Ix*Iy*tdotx*tdoty*sin(2*tx) - Ix*Iz*tdotx*tdoty*sin(2*tx) + Ix*Iy*tdotx*tdotz*cos(ty) - Ix*Iz*tdotx*tdotz*cos(ty) - 2*Iy**2*tdotz**2*cos(tx)**2*cos(ty)*sin(ty) + 2*Iz**2*tdotz**2*cos(tx)**2*cos(ty)*sin(ty) + 2*Iy**2*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty) - 2*Iz**2*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty) + 2*Ix*Iy*tdotz**2*cos(tx)**2*cos(ty)*sin(ty) - 2*Ix*Iz*tdotz**2*cos(tx)**2*cos(ty)*sin(ty) - 2*Ix*Iy*tdotx*tdotz*cos(tx)**2*cos(ty) + 2*Ix*Iz*tdotx*tdotz*cos(tx)**2*cos(ty) - 2*Ix*Iy*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty) + 2*Ix*Iz*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty)))/(Iy*Iz*cos(ty))
        A[8, 7, :] = (dt**2*(Iz**2*tdoty*tdotz + Iy**2*tdoty*tdotz*cos(tx)**2 - Iz**2*tdoty*tdotz*cos(tx)**2 - Iz**2*tdotx*tdoty*sin(ty) - Ix*Iz*tdoty*tdotz + Iy*tauz*cos(tx)*sin(ty) + Iz*tauy*sin(tx)*sin(ty) - Iy**2*tdotx*tdoty*cos(tx)**2*sin(ty) + Iz**2*tdotx*tdoty*cos(tx)**2*sin(ty) - Ix*Iy*tdoty*tdotz*cos(tx)**2 + Ix*Iz*tdoty*tdotz*cos(tx)**2 + Ix*Iz*tdotx*tdoty*sin(ty) + Iy**2*tdotz**2*cos(tx)*cos(ty)**3*sin(tx) - Iz**2*tdotz**2*cos(tx)*cos(ty)**3*sin(tx) - Ix*Iy*tdotz**2*cos(tx)*cos(ty)**3*sin(tx) + Ix*Iz*tdotz**2*cos(tx)*cos(ty)**3*sin(tx) + Ix*Iy*tdotx*tdoty*cos(tx)**2*sin(ty) - Ix*Iz*tdotx*tdoty*cos(tx)**2*sin(ty)))/(Iy*Iz*cos(ty)**2)
        A[8, 8, :] = 1
        A[8, 9, :] = -(dt**2*(Iy**2*tdoty*cos(tx)**2 + Iz**2*tdoty*sin(tx)**2 - Ix*Iy*tdoty*cos(tx)**2 - Ix*Iz*tdoty*sin(tx)**2 + Iy**2*tdotz*cos(tx)*cos(ty)*sin(tx) - Iz**2*tdotz*cos(tx)*cos(ty)*sin(tx) - Ix*Iy*tdotz*cos(tx)*cos(ty)*sin(tx) + Ix*Iz*tdotz*cos(tx)*cos(ty)*sin(tx)))/(Iy*Iz*cos(ty))
        A[8, 10, :] = -(dt**2*(tdotx - tdotz*sin(ty))*(Iy**2*cos(tx)**2 + Iz**2*sin(tx)**2 - Ix*Iy*cos(tx)**2 - Ix*Iz*sin(tx)**2))/(Iy*Iz*cos(ty))
        A[8, 11, :] = (dt*(Iy*Iz*cos(tx)**2*cos(ty) + Iy*Iz*cos(ty)*sin(tx)**2 + Iy**2*dt*tdoty*cos(tx)**2*sin(ty) + Iz**2*dt*tdoty*sin(tx)**2*sin(ty) - Iy**2*dt*tdotx*cos(tx)*cos(ty)*sin(tx) + Iz**2*dt*tdotx*cos(tx)*cos(ty)*sin(tx) - Ix*Iy*dt*tdoty*cos(tx)**2*sin(ty) - Ix*Iz*dt*tdoty*sin(tx)**2*sin(ty) + 2*Iy**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) - 2*Iz**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) + Ix*Iy*dt*tdotx*cos(tx)*cos(ty)*sin(tx) - Ix*Iz*dt*tdotx*cos(tx)*cos(ty)*sin(tx) - 2*Ix*Iy*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) + 2*Ix*Iz*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty)))/(Iy*Iz*cos(ty))
        A[9, 6, :] = (Iy*Iz**2*dt*tdotz**2*cos(ty)**3*sin(tx)**2 - Iy**2*Iz*dt*tdotz**2*cos(ty)**3*sin(tx)**2 + Ix*Iz*dt*tauy*cos(tx)*sin(ty) + Iy*Iz**2*dt*tdoty**2*cos(tx)**2*cos(ty) - Iy**2*Iz*dt*tdoty**2*cos(tx)**2*cos(ty) - Ix*Iy*dt*tauz*sin(tx)*sin(ty) - Iy*Iz**2*dt*tdoty**2*cos(ty)*sin(tx)**2 + Iy**2*Iz*dt*tdoty**2*cos(ty)*sin(tx)**2 - Iy*Iz**2*dt*tdotz**2*cos(tx)**2*cos(ty)**3 + Iy**2*Iz*dt*tdotz**2*cos(tx)**2*cos(ty)**3 + Ix*Iy**2*dt*tdotz**2*cos(tx)**2*cos(ty)*sin(ty)**2 - Ix**2*Iy*dt*tdotz**2*cos(tx)**2*cos(ty)*sin(ty)**2 - Ix*Iz**2*dt*tdotz**2*cos(tx)**2*cos(ty)*sin(ty)**2 + Ix**2*Iz*dt*tdotz**2*cos(tx)**2*cos(ty)*sin(ty)**2 - Ix*Iy**2*dt*tdotz**2*cos(ty)*sin(tx)**2*sin(ty)**2 + Ix**2*Iy*dt*tdotz**2*cos(ty)*sin(tx)**2*sin(ty)**2 + Ix*Iz**2*dt*tdotz**2*cos(ty)*sin(tx)**2*sin(ty)**2 - Ix**2*Iz*dt*tdotz**2*cos(ty)*sin(tx)**2*sin(ty)**2 + 2*Ix*Iy**2*dt*tdotx*tdoty*cos(tx)*sin(tx)*sin(ty) - 2*Ix**2*Iy*dt*tdotx*tdoty*cos(tx)*sin(tx)*sin(ty) - 2*Ix*Iz**2*dt*tdotx*tdoty*cos(tx)*sin(tx)*sin(ty) + 2*Ix**2*Iz*dt*tdotx*tdoty*cos(tx)*sin(tx)*sin(ty) - Ix*Iy**2*dt*tdotx*tdotz*cos(tx)**2*cos(ty)*sin(ty) + Ix**2*Iy*dt*tdotx*tdotz*cos(tx)**2*cos(ty)*sin(ty) + Ix*Iz**2*dt*tdotx*tdotz*cos(tx)**2*cos(ty)*sin(ty) - Ix**2*Iz*dt*tdotx*tdotz*cos(tx)**2*cos(ty)*sin(ty) + 4*Iy*Iz**2*dt*tdoty*tdotz*cos(tx)*cos(ty)**2*sin(tx) - 4*Iy**2*Iz*dt*tdoty*tdotz*cos(tx)*cos(ty)**2*sin(tx) - 2*Ix*Iy**2*dt*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty)**2 + 2*Ix**2*Iy*dt*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty)**2 + Ix*Iy**2*dt*tdotx*tdotz*cos(ty)*sin(tx)**2*sin(ty) - Ix**2*Iy*dt*tdotx*tdotz*cos(ty)*sin(tx)**2*sin(ty) + 2*Ix*Iz**2*dt*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty)**2 - 2*Ix**2*Iz*dt*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty)**2 - Ix*Iz**2*dt*tdotx*tdotz*cos(ty)*sin(tx)**2*sin(ty) + Ix**2*Iz*dt*tdotx*tdotz*cos(ty)*sin(tx)**2*sin(ty))/(Ix*Iy*Iz*cos(ty))
        A[9, 7, :] = (Ix**2*Iz*dt*tdotx*tdoty - Ix*Iz**2*dt*tdotx*tdoty + Ix*Iy*dt*tauz*cos(tx) + Ix*Iz*dt*tauy*sin(tx) + Ix*Iz**2*dt*tdoty*tdotz*sin(ty) - Ix**2*Iz*dt*tdoty*tdotz*sin(ty) - Ix*Iy**2*dt*tdotx*tdoty*cos(tx)**2 + Ix**2*Iy*dt*tdotx*tdoty*cos(tx)**2 + Ix*Iz**2*dt*tdotx*tdoty*cos(tx)**2 - Ix**2*Iz*dt*tdotx*tdoty*cos(tx)**2 + Ix*Iy**2*dt*tdoty*tdotz*cos(tx)**2*sin(ty) - Ix**2*Iy*dt*tdoty*tdotz*cos(tx)**2*sin(ty) - Ix*Iz**2*dt*tdoty*tdotz*cos(tx)**2*sin(ty) + Ix**2*Iz*dt*tdoty*tdotz*cos(tx)**2*sin(ty) + Ix*Iz**2*dt*tdoty*tdotz*cos(ty)**2*sin(ty) - Ix**2*Iz*dt*tdoty*tdotz*cos(ty)**2*sin(ty) - Iy*Iz**2*dt*tdoty*tdotz*cos(ty)**2*sin(ty) + Iy**2*Iz*dt*tdoty*tdotz*cos(ty)**2*sin(ty) + 2*Ix*Iy**2*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) - 2*Ix**2*Iy*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) - 2*Ix*Iz**2*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) + 2*Ix**2*Iz*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) + 2*Iy*Iz**2*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) - 2*Iy**2*Iz*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) - Ix*Iy**2*dt*tdotx*tdotz*cos(tx)*cos(ty)**3*sin(tx) + Ix**2*Iy*dt*tdotx*tdotz*cos(tx)*cos(ty)**3*sin(tx) + Ix*Iz**2*dt*tdotx*tdotz*cos(tx)*cos(ty)**3*sin(tx) - Ix**2*Iz*dt*tdotx*tdotz*cos(tx)*cos(ty)**3*sin(tx) + Ix*Iy**2*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty) - Ix**2*Iy*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty) - Ix*Iz**2*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty) + Ix**2*Iz*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty) + 2*Iy*Iz**2*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty) - 2*Iy**2*Iz*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty))/(Ix*Iy*Iz*cos(ty)**2)
        A[9, 9, :] = (Iy*Iz*cos(ty) - Iy**2*dt*tdoty*cos(tx)**2*sin(ty) - Iz**2*dt*tdoty*sin(tx)**2*sin(ty) + Ix*Iy*dt*tdoty*cos(tx)**2*sin(ty) + Ix*Iz*dt*tdoty*sin(tx)**2*sin(ty) - Iy**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) + Iz**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) + Ix*Iy*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) - Ix*Iz*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty))/(Iy*Iz*cos(ty))
        A[9, 10, :] = -(dt*(Iy*Iz**2*tdotz*cos(tx)**2*cos(ty)**2 - Iy**2*Iz*tdotz*cos(tx)**2*cos(ty)**2 - Ix*Iy**2*tdotz*cos(tx)**2*sin(ty)**2 + Ix**2*Iy*tdotz*cos(tx)**2*sin(ty)**2 - Iy*Iz**2*tdotz*cos(ty)**2*sin(tx)**2 + Iy**2*Iz*tdotz*cos(ty)**2*sin(tx)**2 - Ix*Iz**2*tdotz*sin(tx)**2*sin(ty)**2 + Ix**2*Iz*tdotz*sin(tx)**2*sin(ty)**2 + Ix*Iy**2*tdotx*cos(tx)**2*sin(ty) - Ix**2*Iy*tdotx*cos(tx)**2*sin(ty) + Ix*Iz**2*tdotx*sin(tx)**2*sin(ty) - Ix**2*Iz*tdotx*sin(tx)**2*sin(ty) - 2*Iy*Iz**2*tdoty*cos(tx)*cos(ty)*sin(tx) + 2*Iy**2*Iz*tdoty*cos(tx)*cos(ty)*sin(tx)))/(Ix*Iy*Iz*cos(ty))
        A[9, 11, :] = (Ix*Iy*Iz*cos(tx)**2*cos(ty)*sin(ty) - (Ix*Iy*Iz*sin(2*ty))/2 + Ix*Iy*Iz*cos(ty)*sin(tx)**2*sin(ty) - Iy*Iz**2*dt*tdoty*cos(tx)**2*cos(ty)**2 + Iy**2*Iz*dt*tdoty*cos(tx)**2*cos(ty)**2 + Ix*Iy**2*dt*tdoty*cos(tx)**2*sin(ty)**2 - Ix**2*Iy*dt*tdoty*cos(tx)**2*sin(ty)**2 + Iy*Iz**2*dt*tdoty*cos(ty)**2*sin(tx)**2 - Iy**2*Iz*dt*tdoty*cos(ty)**2*sin(tx)**2 + Ix*Iz**2*dt*tdoty*sin(tx)**2*sin(ty)**2 - Ix**2*Iz*dt*tdoty*sin(tx)**2*sin(ty)**2 - 2*Iy*Iz**2*dt*tdotz*cos(tx)*cos(ty)**3*sin(tx) + 2*Iy**2*Iz*dt*tdotz*cos(tx)*cos(ty)**3*sin(tx) + 2*Ix*Iy**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty)**2 - 2*Ix**2*Iy*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty)**2 - 2*Ix*Iz**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty)**2 + 2*Ix**2*Iz*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty)**2 - Ix*Iy**2*dt*tdotx*cos(tx)*cos(ty)*sin(tx)*sin(ty) + Ix**2*Iy*dt*tdotx*cos(tx)*cos(ty)*sin(tx)*sin(ty) + Ix*Iz**2*dt*tdotx*cos(tx)*cos(ty)*sin(tx)*sin(ty) - Ix**2*Iz*dt*tdotx*cos(tx)*cos(ty)*sin(tx)*sin(ty))/(Ix*Iy*Iz*cos(ty))
        A[10, 6, :] = -(dt*(Iz*tauy*sin(tx) + Iy**2*tdotx*tdoty - Iz**2*tdotx*tdoty + Iy*tauz*cos(tx) - 2*Iy**2*tdotx*tdoty*cos(tx)**2 + 2*Iz**2*tdotx*tdoty*cos(tx)**2 - Iy**2*tdoty*tdotz*sin(ty) + Iz**2*tdoty*tdotz*sin(ty) - Ix*Iy*tdotx*tdoty + Ix*Iz*tdotx*tdoty + 2*Iy**2*tdoty*tdotz*cos(tx)**2*sin(ty) - 2*Iz**2*tdoty*tdotz*cos(tx)**2*sin(ty) + 2*Ix*Iy*tdotx*tdoty*cos(tx)**2 - 2*Ix*Iz*tdotx*tdoty*cos(tx)**2 + Ix*Iy*tdoty*tdotz*sin(ty) - Ix*Iz*tdoty*tdotz*sin(ty) + 2*Iy**2*tdotz**2*cos(tx)*cos(ty)*sin(tx)*sin(ty) - 2*Iz**2*tdotz**2*cos(tx)*cos(ty)*sin(tx)*sin(ty) - 2*Iy**2*tdotx*tdotz*cos(tx)*cos(ty)*sin(tx) + 2*Iz**2*tdotx*tdotz*cos(tx)*cos(ty)*sin(tx) - 2*Ix*Iy*tdoty*tdotz*cos(tx)**2*sin(ty) + 2*Ix*Iz*tdoty*tdotz*cos(tx)**2*sin(ty) - 2*Ix*Iy*tdotz**2*cos(tx)*cos(ty)*sin(tx)*sin(ty) + 2*Ix*Iz*tdotz**2*cos(tx)*cos(ty)*sin(tx)*sin(ty) + 2*Ix*Iy*tdotx*tdotz*cos(tx)*cos(ty)*sin(tx) - 2*Ix*Iz*tdotx*tdotz*cos(tx)*cos(ty)*sin(tx)))/(Iy*Iz)
        A[10, 7, :] = (dt*tdotz*(Iy**2*tdotz*sin(tx)**2*sin(ty)**2 - Iz**2*tdotx*cos(tx)**2*sin(ty) - Iy**2*tdotx*sin(tx)**2*sin(ty) - Iz**2*tdotz*cos(tx)**2*cos(ty)**2 - Iy**2*tdotz*cos(ty)**2*sin(tx)**2 + Iz**2*tdotz*cos(tx)**2*sin(ty)**2 - Iy**2*tdoty*cos(tx)*cos(ty)*sin(tx) + Iz**2*tdoty*cos(tx)*cos(ty)*sin(tx) + Ix*Iz*tdotx*cos(tx)**2*sin(ty) + Ix*Iy*tdotx*sin(tx)**2*sin(ty) + Ix*Iz*tdotz*cos(tx)**2*cos(ty)**2 + Ix*Iy*tdotz*cos(ty)**2*sin(tx)**2 - Ix*Iz*tdotz*cos(tx)**2*sin(ty)**2 - Ix*Iy*tdotz*sin(tx)**2*sin(ty)**2 + Ix*Iy*tdoty*cos(tx)*cos(ty)*sin(tx) - Ix*Iz*tdoty*cos(tx)*cos(ty)*sin(tx)))/(Iy*Iz)
        A[10, 9, :] = (dt*((Iy**2*tdoty*sin(2*tx))/2 - (Iz**2*tdoty*sin(2*tx))/2 + Iz**2*tdotz*cos(tx)**2*cos(ty) + Iy**2*tdotz*cos(ty)*sin(tx)**2 - (Ix*Iy*tdoty*sin(2*tx))/2 + (Ix*Iz*tdoty*sin(2*tx))/2 - Ix*Iz*tdotz*cos(tx)**2*cos(ty) - Ix*Iy*tdotz*cos(ty)*sin(tx)**2))/(Iy*Iz)
        A[10, 10, :] = (2*Iy*Iz + Iy**2*dt*tdotx*sin(2*tx) - Iz**2*dt*tdotx*sin(2*tx) - Ix*Iy*dt*tdotx*sin(2*tx) + Ix*Iz*dt*tdotx*sin(2*tx) - 2*Iy**2*dt*tdotz*cos(tx)*sin(tx)*sin(ty) + 2*Iz**2*dt*tdotz*cos(tx)*sin(tx)*sin(ty) + 2*Ix*Iy*dt*tdotz*cos(tx)*sin(tx)*sin(ty) - 2*Ix*Iz*dt*tdotz*cos(tx)*sin(tx)*sin(ty))/(2*Iy*Iz)
        A[10, 11, :] = (dt*(Iz**2*tdotx*cos(tx)**2*cos(ty) + Iy**2*tdotx*cos(ty)*sin(tx)**2 - Iy**2*tdoty*cos(tx)*sin(tx)*sin(ty) + Iz**2*tdoty*cos(tx)*sin(tx)*sin(ty) - 2*Iz**2*tdotz*cos(tx)**2*cos(ty)*sin(ty) - Ix*Iz*tdotx*cos(tx)**2*cos(ty) - 2*Iy**2*tdotz*cos(ty)*sin(tx)**2*sin(ty) - Ix*Iy*tdotx*cos(ty)*sin(tx)**2 + Ix*Iy*tdoty*cos(tx)*sin(tx)*sin(ty) - Ix*Iz*tdoty*cos(tx)*sin(tx)*sin(ty) + 2*Ix*Iz*tdotz*cos(tx)**2*cos(ty)*sin(ty) + 2*Ix*Iy*tdotz*cos(ty)*sin(tx)**2*sin(ty)))/(Iy*Iz)
        A[11, 6, :] = -(dt*(Iy*tauz*sin(tx) + (Iy**2*tdotz**2*sin(2*ty))/2 - (Iz**2*tdotz**2*sin(2*ty))/2 - Iz*tauy*cos(tx) - (Ix*Iy*tdotz**2*sin(2*ty))/2 + (Ix*Iz*tdotz**2*sin(2*ty))/2 - Iy**2*tdotx*tdoty*sin(2*tx) + Iz**2*tdotx*tdoty*sin(2*tx) - Iy**2*tdotx*tdotz*cos(ty) + Iz**2*tdotx*tdotz*cos(ty) + 2*Iy**2*tdotx*tdotz*cos(tx)**2*cos(ty) - 2*Iz**2*tdotx*tdotz*cos(tx)**2*cos(ty) + Ix*Iy*tdotx*tdoty*sin(2*tx) - Ix*Iz*tdotx*tdoty*sin(2*tx) + Ix*Iy*tdotx*tdotz*cos(ty) - Ix*Iz*tdotx*tdotz*cos(ty) - 2*Iy**2*tdotz**2*cos(tx)**2*cos(ty)*sin(ty) + 2*Iz**2*tdotz**2*cos(tx)**2*cos(ty)*sin(ty) + 2*Iy**2*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty) - 2*Iz**2*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty) + 2*Ix*Iy*tdotz**2*cos(tx)**2*cos(ty)*sin(ty) - 2*Ix*Iz*tdotz**2*cos(tx)**2*cos(ty)*sin(ty) - 2*Ix*Iy*tdotx*tdotz*cos(tx)**2*cos(ty) + 2*Ix*Iz*tdotx*tdotz*cos(tx)**2*cos(ty) - 2*Ix*Iy*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty) + 2*Ix*Iz*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty)))/(Iy*Iz*cos(ty))
        A[11, 7, :] = (dt*(Iz**2*tdoty*tdotz + Iy**2*tdoty*tdotz*cos(tx)**2 - Iz**2*tdoty*tdotz*cos(tx)**2 - Iz**2*tdotx*tdoty*sin(ty) - Ix*Iz*tdoty*tdotz + Iy*tauz*cos(tx)*sin(ty) + Iz*tauy*sin(tx)*sin(ty) - Iy**2*tdotx*tdoty*cos(tx)**2*sin(ty) + Iz**2*tdotx*tdoty*cos(tx)**2*sin(ty) - Ix*Iy*tdoty*tdotz*cos(tx)**2 + Ix*Iz*tdoty*tdotz*cos(tx)**2 + Ix*Iz*tdotx*tdoty*sin(ty) + Iy**2*tdotz**2*cos(tx)*cos(ty)**3*sin(tx) - Iz**2*tdotz**2*cos(tx)*cos(ty)**3*sin(tx) - Ix*Iy*tdotz**2*cos(tx)*cos(ty)**3*sin(tx) + Ix*Iz*tdotz**2*cos(tx)*cos(ty)**3*sin(tx) + Ix*Iy*tdotx*tdoty*cos(tx)**2*sin(ty) - Ix*Iz*tdotx*tdoty*cos(tx)**2*sin(ty)))/(Iy*Iz*cos(ty)**2)
        A[11, 9, :] = -(dt*(Iy**2*tdoty*cos(tx)**2 + Iz**2*tdoty*sin(tx)**2 - Ix*Iy*tdoty*cos(tx)**2 - Ix*Iz*tdoty*sin(tx)**2 + Iy**2*tdotz*cos(tx)*cos(ty)*sin(tx) - Iz**2*tdotz*cos(tx)*cos(ty)*sin(tx) - Ix*Iy*tdotz*cos(tx)*cos(ty)*sin(tx) + Ix*Iz*tdotz*cos(tx)*cos(ty)*sin(tx)))/(Iy*Iz*cos(ty))
        A[11, 10, :] = -(dt*(tdotx - tdotz*sin(ty))*(Iy**2*cos(tx)**2 + Iz**2*sin(tx)**2 - Ix*Iy*cos(tx)**2 - Ix*Iz*sin(tx)**2))/(Iy*Iz*cos(ty))
        A[11, 11, :] = (Iy*Iz*cos(tx)**2*cos(ty) + Iy*Iz*cos(ty)*sin(tx)**2 + Iy**2*dt*tdoty*cos(tx)**2*sin(ty) + Iz**2*dt*tdoty*sin(tx)**2*sin(ty) - Iy**2*dt*tdotx*cos(tx)*cos(ty)*sin(tx) + Iz**2*dt*tdotx*cos(tx)*cos(ty)*sin(tx) - Ix*Iy*dt*tdoty*cos(tx)**2*sin(ty) - Ix*Iz*dt*tdoty*sin(tx)**2*sin(ty) + 2*Iy**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) - 2*Iz**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) + Ix*Iy*dt*tdotx*cos(tx)*cos(ty)*sin(tx) - Ix*Iz*dt*tdotx*cos(tx)*cos(ty)*sin(tx) - 2*Ix*Iy*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) + 2*Ix*Iz*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty))/(Iy*Iz*cos(ty))
        return A
    
    def get_A_matrix(self, x, u, horizon):
        A = np.zeros((self.Dx, self.Dx, horizon))
        tx, ty, tz = x[6], x[7], x[8]
        tdotx, tdoty, tdotz = x[9], x[10], x[11]
        Tau, taux, tauy, tauz = u[0], u[1], u[2], u[3]
        self.lib.compute_A_matrix(
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tdotx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tdoty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tdotz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Tau.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            taux.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tauy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tauz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            horizon, self.I[0,0], self.I[1,1], self.I[2,2], 
            self.kd, self.m, self.dt
        )
        return A
    
    # def get_B_matrix(self, x, u, horizon, inc = 0.01):
    #     B = np.zeros((self.Dx, self.Du, horizon))
    #     for t in range(horizon):
    #         for i in range(self.Du):
    #             up, um = u[:,t].copy(), u[:,t].copy()
    #             up[i] += inc
    #             xnextp = self.forward_simulate(x[:,t], up)
    #             um[i] -= inc
    #             xnextm = self.forward_simulate(x[:,t], um)
    #             diff = (xnextp - xnextm)/(2*inc)
    #             B[:,i,t] = diff
    #     # print(x)
    #     return B

    def get_B_matrix_py(self, x, u, horizon):
        B = np.zeros((self.Dx, self.Du, horizon), dtype=np.float64)
        Ix = self.I[0, 0]; Iy = self.I[1, 1]; Iz = self.I[2, 2]
        m = self.m; dt = self.dt; kd = self.kd
        tx, ty, tz = x[6], x[7], x[8]
        B[3, 0, :] = (dt*sin(ty)*sin(tz))/m
        B[4, 0, :] = -(dt*cos(tz)*sin(ty))/m
        B[5, 0, :] = (dt*cos(ty))/m
        B[6, 1, :] = dt**2/Ix
        B[6, 2, :] = (dt**2*sin(tx)*sin(ty))/(Iy*cos(ty))
        B[6, 3, :] = (dt**2*cos(tx)*sin(ty))/(Iz*cos(ty))
        B[7, 2, :] = (dt**2*cos(tx))/Iy
        B[7, 3, :] = -(dt**2*sin(tx))/Iz
        B[8, 2, :] = (dt**2*sin(tx))/(Iy*cos(ty))
        B[8, 3, :] = (dt**2*cos(tx))/(Iz*cos(ty))
        B[9, 1, :] = dt/Ix
        B[9, 2, :] = (dt*sin(tx)*sin(ty))/(Iy*cos(ty))
        B[9, 3, :] = (dt*cos(tx)*sin(ty))/(Iz*cos(ty))
        B[10, 2, :] = (dt*cos(tx))/Iy
        B[10, 3, :] = -(dt*sin(tx))/Iz
        B[11, 2, :] = (dt*sin(tx))/(Iy*cos(ty))
        B[11, 3, :] = (dt*cos(tx))/(Iz*cos(ty))
        return B
    
    def get_B_matrix(self, x, u, horizon):
        x = x.reshape(self.Dx, -1); u = u.reshape(self.Du, -1)
        B = np.zeros((self.Dx, self.Du, horizon), dtype=np.float64)
        tx, ty, tz = x[6], x[7], x[8] # vector!!!
        self.lib.compute_B_matrix(
            B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            horizon, self.I[0, 0], self.I[1, 1], self.I[2, 2], 
            self.m, self.dt
            )
        return B

    def forward_simulate_py(self, x, u):
        Ix = self.I[0, 0]; Iy = self.I[1, 1]; Iz = self.I[2, 2]
        m = self.m; dt = self.dt; kd = self.kd
        tx, ty, tz = x[6], x[7], x[8]
        p, pdot, theta_thetadot = x[:3], x[3:6], x[6:]
        tx, ty, tz = theta_thetadot[0], theta_thetadot[1], theta_thetadot[2]
        tdotx, tdoty, tdotz = theta_thetadot[3], theta_thetadot[4], theta_thetadot[5]
        p_next, pdot_next, theta_thetadot_next = np.zeros(3), np.zeros(3), np.zeros(6)
        Tau, taux, tauy, tauz = u[0], u[1], u[2], u[3]

        p_next = p + dt * pdot
        pdot_next[0] = (1-dt*kd)*pdot[0]+x[7]*(Tau*dt*cos(ty)*sin(tz))/m+x[8]*(Tau*dt*cos(tz)*sin(ty))/m
        pdot_next[1] = (1-dt*kd)*pdot[1]-x[7]*(Tau*dt*cos(ty)*cos(tz))/m+x[8]*(Tau*dt*sin(ty)*sin(tz))/m
        pdot_next[2] = (1-dt*kd)*pdot[2]-x[7]*(Tau*dt*sin(ty))/m

        theta_A = np.zeros((6, 6))
        theta_B = np.zeros((6, 3))
        theta_A[0, 0] = (Ix*Iy*Iz*cos(ty) - Iy*Iz**2*dt**2*tdoty**2*cos(ty)*sin(tx)**2 + Iy**2*Iz*dt**2*tdoty**2*cos(ty)*sin(tx)**2 - Iy*Iz**2*dt**2*tdotz**2*cos(tx)**2*cos(ty)**3 + Iy**2*Iz*dt**2*tdotz**2*cos(tx)**2*cos(ty)**3 + Iy*Iz**2*dt**2*tdotz**2*cos(ty)**3*sin(tx)**2 - Iy**2*Iz*dt**2*tdotz**2*cos(ty)**3*sin(tx)**2 + Ix*Iz*dt**2*tauy*cos(tx)*sin(ty) + Iy*Iz**2*dt**2*tdoty**2*cos(tx)**2*cos(ty) - Iy**2*Iz*dt**2*tdoty**2*cos(tx)**2*cos(ty) - Ix*Iy*dt**2*tauz*sin(tx)*sin(ty) + Ix*Iy**2*dt**2*tdotz**2*cos(tx)**2*cos(ty)*sin(ty)**2 - Ix**2*Iy*dt**2*tdotz**2*cos(tx)**2*cos(ty)*sin(ty)**2 - Ix*Iz**2*dt**2*tdotz**2*cos(tx)**2*cos(ty)*sin(ty)**2 + Ix**2*Iz*dt**2*tdotz**2*cos(tx)**2*cos(ty)*sin(ty)**2 - Ix*Iy**2*dt**2*tdotz**2*cos(ty)*sin(tx)**2*sin(ty)**2 + Ix**2*Iy*dt**2*tdotz**2*cos(ty)*sin(tx)**2*sin(ty)**2 + Ix*Iz**2*dt**2*tdotz**2*cos(ty)*sin(tx)**2*sin(ty)**2 - Ix**2*Iz*dt**2*tdotz**2*cos(ty)*sin(tx)**2*sin(ty)**2 - 2*Ix*Iy**2*dt**2*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty)**2 + 2*Ix**2*Iy*dt**2*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty)**2 + Ix*Iy**2*dt**2*tdotx*tdotz*cos(ty)*sin(tx)**2*sin(ty) - Ix**2*Iy*dt**2*tdotx*tdotz*cos(ty)*sin(tx)**2*sin(ty) + 2*Ix*Iz**2*dt**2*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty)**2 - 2*Ix**2*Iz*dt**2*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty)**2 - Ix*Iz**2*dt**2*tdotx*tdotz*cos(ty)*sin(tx)**2*sin(ty) + Ix**2*Iz*dt**2*tdotx*tdotz*cos(ty)*sin(tx)**2*sin(ty) + 2*Ix*Iy**2*dt**2*tdotx*tdoty*cos(tx)*sin(tx)*sin(ty) - 2*Ix**2*Iy*dt**2*tdotx*tdoty*cos(tx)*sin(tx)*sin(ty) - 2*Ix*Iz**2*dt**2*tdotx*tdoty*cos(tx)*sin(tx)*sin(ty) + 2*Ix**2*Iz*dt**2*tdotx*tdoty*cos(tx)*sin(tx)*sin(ty) - Ix*Iy**2*dt**2*tdotx*tdotz*cos(tx)**2*cos(ty)*sin(ty) + Ix**2*Iy*dt**2*tdotx*tdotz*cos(tx)**2*cos(ty)*sin(ty) + Ix*Iz**2*dt**2*tdotx*tdotz*cos(tx)**2*cos(ty)*sin(ty) - Ix**2*Iz*dt**2*tdotx*tdotz*cos(tx)**2*cos(ty)*sin(ty) + 4*Iy*Iz**2*dt**2*tdoty*tdotz*cos(tx)*cos(ty)**2*sin(tx) - 4*Iy**2*Iz*dt**2*tdoty*tdotz*cos(tx)*cos(ty)**2*sin(tx))/(Ix*Iy*Iz*cos(ty))
        theta_A[0, 1] = (dt*(Ix**2*Iz*dt*tdotx*tdoty - Ix*Iz**2*dt*tdotx*tdoty + Ix*Iy*dt*tauz*cos(tx) + Ix*Iz*dt*tauy*sin(tx) + Ix*Iz**2*dt*tdoty*tdotz*sin(ty) - Ix**2*Iz*dt*tdoty*tdotz*sin(ty) - Ix*Iy**2*dt*tdotx*tdoty*cos(tx)**2 + Ix**2*Iy*dt*tdotx*tdoty*cos(tx)**2 + Ix*Iz**2*dt*tdotx*tdoty*cos(tx)**2 - Ix**2*Iz*dt*tdotx*tdoty*cos(tx)**2 + Ix*Iy**2*dt*tdoty*tdotz*cos(tx)**2*sin(ty) - Ix**2*Iy*dt*tdoty*tdotz*cos(tx)**2*sin(ty) - Ix*Iz**2*dt*tdoty*tdotz*cos(tx)**2*sin(ty) + Ix**2*Iz*dt*tdoty*tdotz*cos(tx)**2*sin(ty) + Ix*Iz**2*dt*tdoty*tdotz*cos(ty)**2*sin(ty) - Ix**2*Iz*dt*tdoty*tdotz*cos(ty)**2*sin(ty) - Iy*Iz**2*dt*tdoty*tdotz*cos(ty)**2*sin(ty) + Iy**2*Iz*dt*tdoty*tdotz*cos(ty)**2*sin(ty) + 2*Ix*Iy**2*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) - 2*Ix**2*Iy*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) - 2*Ix*Iz**2*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) + 2*Ix**2*Iz*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) + 2*Iy*Iz**2*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) - 2*Iy**2*Iz*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) - Ix*Iy**2*dt*tdotx*tdotz*cos(tx)*cos(ty)**3*sin(tx) + Ix**2*Iy*dt*tdotx*tdotz*cos(tx)*cos(ty)**3*sin(tx) + Ix*Iz**2*dt*tdotx*tdotz*cos(tx)*cos(ty)**3*sin(tx) - Ix**2*Iz*dt*tdotx*tdotz*cos(tx)*cos(ty)**3*sin(tx) + Ix*Iy**2*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty) - Ix**2*Iy*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty) - Ix*Iz**2*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty) + Ix**2*Iz*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty) + 2*Iy*Iz**2*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty) - 2*Iy**2*Iz*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty)))/(Ix*Iy*Iz*cos(ty)**2)
        theta_A[0, 3] = (dt*(Iy*Iz*cos(ty) - Iy**2*dt*tdoty*cos(tx)**2*sin(ty) - Iz**2*dt*tdoty*sin(tx)**2*sin(ty) + Ix*Iy*dt*tdoty*cos(tx)**2*sin(ty) + Ix*Iz*dt*tdoty*sin(tx)**2*sin(ty) - Iy**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) + Iz**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) + Ix*Iy*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) - Ix*Iz*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty)))/(Iy*Iz*cos(ty))
        theta_A[0, 4] = -(dt**2*(Iy*Iz**2*tdotz*cos(tx)**2*cos(ty)**2 - Iy**2*Iz*tdotz*cos(tx)**2*cos(ty)**2 - Ix*Iy**2*tdotz*cos(tx)**2*sin(ty)**2 + Ix**2*Iy*tdotz*cos(tx)**2*sin(ty)**2 - Iy*Iz**2*tdotz*cos(ty)**2*sin(tx)**2 + Iy**2*Iz*tdotz*cos(ty)**2*sin(tx)**2 - Ix*Iz**2*tdotz*sin(tx)**2*sin(ty)**2 + Ix**2*Iz*tdotz*sin(tx)**2*sin(ty)**2 + Ix*Iy**2*tdotx*cos(tx)**2*sin(ty) - Ix**2*Iy*tdotx*cos(tx)**2*sin(ty) + Ix*Iz**2*tdotx*sin(tx)**2*sin(ty) - Ix**2*Iz*tdotx*sin(tx)**2*sin(ty) - 2*Iy*Iz**2*tdoty*cos(tx)*cos(ty)*sin(tx) + 2*Iy**2*Iz*tdoty*cos(tx)*cos(ty)*sin(tx)))/(Ix*Iy*Iz*cos(ty))
        theta_A[0, 5] = (dt*(Ix*Iy*Iz*cos(tx)**2*cos(ty)*sin(ty) - (Ix*Iy*Iz*sin(2*ty))/2 + Ix*Iy*Iz*cos(ty)*sin(tx)**2*sin(ty) - Iy*Iz**2*dt*tdoty*cos(tx)**2*cos(ty)**2 + Iy**2*Iz*dt*tdoty*cos(tx)**2*cos(ty)**2 + Ix*Iy**2*dt*tdoty*cos(tx)**2*sin(ty)**2 - Ix**2*Iy*dt*tdoty*cos(tx)**2*sin(ty)**2 + Iy*Iz**2*dt*tdoty*cos(ty)**2*sin(tx)**2 - Iy**2*Iz*dt*tdoty*cos(ty)**2*sin(tx)**2 + Ix*Iz**2*dt*tdoty*sin(tx)**2*sin(ty)**2 - Ix**2*Iz*dt*tdoty*sin(tx)**2*sin(ty)**2 - 2*Iy*Iz**2*dt*tdotz*cos(tx)*cos(ty)**3*sin(tx) + 2*Iy**2*Iz*dt*tdotz*cos(tx)*cos(ty)**3*sin(tx) + 2*Ix*Iy**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty)**2 - 2*Ix**2*Iy*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty)**2 - 2*Ix*Iz**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty)**2 + 2*Ix**2*Iz*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty)**2 - Ix*Iy**2*dt*tdotx*cos(tx)*cos(ty)*sin(tx)*sin(ty) + Ix**2*Iy*dt*tdotx*cos(tx)*cos(ty)*sin(tx)*sin(ty) + Ix*Iz**2*dt*tdotx*cos(tx)*cos(ty)*sin(tx)*sin(ty) - Ix**2*Iz*dt*tdotx*cos(tx)*cos(ty)*sin(tx)*sin(ty)))/(Ix*Iy*Iz*cos(ty))
        theta_A[1, 0] = -(dt**2*(Iz*tauy*sin(tx) + Iy**2*tdotx*tdoty - Iz**2*tdotx*tdoty + Iy*tauz*cos(tx) - 2*Iy**2*tdotx*tdoty*cos(tx)**2 + 2*Iz**2*tdotx*tdoty*cos(tx)**2 - Iy**2*tdoty*tdotz*sin(ty) + Iz**2*tdoty*tdotz*sin(ty) - Ix*Iy*tdotx*tdoty + Ix*Iz*tdotx*tdoty + 2*Iy**2*tdoty*tdotz*cos(tx)**2*sin(ty) - 2*Iz**2*tdoty*tdotz*cos(tx)**2*sin(ty) + 2*Ix*Iy*tdotx*tdoty*cos(tx)**2 - 2*Ix*Iz*tdotx*tdoty*cos(tx)**2 + Ix*Iy*tdoty*tdotz*sin(ty) - Ix*Iz*tdoty*tdotz*sin(ty) + 2*Iy**2*tdotz**2*cos(tx)*cos(ty)*sin(tx)*sin(ty) - 2*Iz**2*tdotz**2*cos(tx)*cos(ty)*sin(tx)*sin(ty) - 2*Iy**2*tdotx*tdotz*cos(tx)*cos(ty)*sin(tx) + 2*Iz**2*tdotx*tdotz*cos(tx)*cos(ty)*sin(tx) - 2*Ix*Iy*tdoty*tdotz*cos(tx)**2*sin(ty) + 2*Ix*Iz*tdoty*tdotz*cos(tx)**2*sin(ty) - 2*Ix*Iy*tdotz**2*cos(tx)*cos(ty)*sin(tx)*sin(ty) + 2*Ix*Iz*tdotz**2*cos(tx)*cos(ty)*sin(tx)*sin(ty) + 2*Ix*Iy*tdotx*tdotz*cos(tx)*cos(ty)*sin(tx) - 2*Ix*Iz*tdotx*tdotz*cos(tx)*cos(ty)*sin(tx)))/(Iy*Iz)
        theta_A[1, 1] = (Iy*Iz - Iz**2*dt**2*tdotz**2*cos(tx)**2*cos(ty)**2 - Iy**2*dt**2*tdotz**2*cos(ty)**2*sin(tx)**2 + Iz**2*dt**2*tdotz**2*cos(tx)**2*sin(ty)**2 + Iy**2*dt**2*tdotz**2*sin(tx)**2*sin(ty)**2 + Ix*Iy*dt**2*tdotz**2*cos(ty)**2*sin(tx)**2 - Ix*Iz*dt**2*tdotz**2*cos(tx)**2*sin(ty)**2 - Ix*Iy*dt**2*tdotz**2*sin(tx)**2*sin(ty)**2 - Iz**2*dt**2*tdotx*tdotz*cos(tx)**2*sin(ty) - Iy**2*dt**2*tdotx*tdotz*sin(tx)**2*sin(ty) + Ix*Iz*dt**2*tdotz**2*cos(tx)**2*cos(ty)**2 - Iy**2*dt**2*tdoty*tdotz*cos(tx)*cos(ty)*sin(tx) + Iz**2*dt**2*tdoty*tdotz*cos(tx)*cos(ty)*sin(tx) + Ix*Iz*dt**2*tdotx*tdotz*cos(tx)**2*sin(ty) + Ix*Iy*dt**2*tdotx*tdotz*sin(tx)**2*sin(ty) + Ix*Iy*dt**2*tdoty*tdotz*cos(tx)*cos(ty)*sin(tx) - Ix*Iz*dt**2*tdoty*tdotz*cos(tx)*cos(ty)*sin(tx))/(Iy*Iz)
        theta_A[1, 3] = (dt**2*((Iy**2*tdoty*sin(2*tx))/2 - (Iz**2*tdoty*sin(2*tx))/2 + Iz**2*tdotz*cos(tx)**2*cos(ty) + Iy**2*tdotz*cos(ty)*sin(tx)**2 - (Ix*Iy*tdoty*sin(2*tx))/2 + (Ix*Iz*tdoty*sin(2*tx))/2 - Ix*Iz*tdotz*cos(tx)**2*cos(ty) - Ix*Iy*tdotz*cos(ty)*sin(tx)**2))/(Iy*Iz)
        theta_A[1, 4] = (dt*(Iy*Iz + (Iy**2*dt*tdotx*sin(2*tx))/2 - (Iz**2*dt*tdotx*sin(2*tx))/2 - (Ix*Iy*dt*tdotx*sin(2*tx))/2 + (Ix*Iz*dt*tdotx*sin(2*tx))/2 - Iy**2*dt*tdotz*cos(tx)*sin(tx)*sin(ty) + Iz**2*dt*tdotz*cos(tx)*sin(tx)*sin(ty) + Ix*Iy*dt*tdotz*cos(tx)*sin(tx)*sin(ty) - Ix*Iz*dt*tdotz*cos(tx)*sin(tx)*sin(ty)))/(Iy*Iz)
        theta_A[1, 5] = (dt**2*(Iz**2*tdotx*cos(tx)**2*cos(ty) + Iy**2*tdotx*cos(ty)*sin(tx)**2 - Iy**2*tdoty*cos(tx)*sin(tx)*sin(ty) + Iz**2*tdoty*cos(tx)*sin(tx)*sin(ty) - 2*Iz**2*tdotz*cos(tx)**2*cos(ty)*sin(ty) - Ix*Iz*tdotx*cos(tx)**2*cos(ty) - 2*Iy**2*tdotz*cos(ty)*sin(tx)**2*sin(ty) - Ix*Iy*tdotx*cos(ty)*sin(tx)**2 + Ix*Iy*tdoty*cos(tx)*sin(tx)*sin(ty) - Ix*Iz*tdoty*cos(tx)*sin(tx)*sin(ty) + 2*Ix*Iz*tdotz*cos(tx)**2*cos(ty)*sin(ty) + 2*Ix*Iy*tdotz*cos(ty)*sin(tx)**2*sin(ty)))/(Iy*Iz)
        theta_A[2, 0] = -(dt**2*(Iy*tauz*sin(tx) + (Iy**2*tdotz**2*sin(2*ty))/2 - (Iz**2*tdotz**2*sin(2*ty))/2 - Iz*tauy*cos(tx) - (Ix*Iy*tdotz**2*sin(2*ty))/2 + (Ix*Iz*tdotz**2*sin(2*ty))/2 - Iy**2*tdotx*tdoty*sin(2*tx) + Iz**2*tdotx*tdoty*sin(2*tx) - Iy**2*tdotx*tdotz*cos(ty) + Iz**2*tdotx*tdotz*cos(ty) + 2*Iy**2*tdotx*tdotz*cos(tx)**2*cos(ty) - 2*Iz**2*tdotx*tdotz*cos(tx)**2*cos(ty) + Ix*Iy*tdotx*tdoty*sin(2*tx) - Ix*Iz*tdotx*tdoty*sin(2*tx) + Ix*Iy*tdotx*tdotz*cos(ty) - Ix*Iz*tdotx*tdotz*cos(ty) - 2*Iy**2*tdotz**2*cos(tx)**2*cos(ty)*sin(ty) + 2*Iz**2*tdotz**2*cos(tx)**2*cos(ty)*sin(ty) + 2*Iy**2*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty) - 2*Iz**2*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty) + 2*Ix*Iy*tdotz**2*cos(tx)**2*cos(ty)*sin(ty) - 2*Ix*Iz*tdotz**2*cos(tx)**2*cos(ty)*sin(ty) - 2*Ix*Iy*tdotx*tdotz*cos(tx)**2*cos(ty) + 2*Ix*Iz*tdotx*tdotz*cos(tx)**2*cos(ty) - 2*Ix*Iy*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty) + 2*Ix*Iz*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty)))/(Iy*Iz*cos(ty))
        theta_A[2, 1] = (dt**2*(Iz**2*tdoty*tdotz + Iy**2*tdoty*tdotz*cos(tx)**2 - Iz**2*tdoty*tdotz*cos(tx)**2 - Iz**2*tdotx*tdoty*sin(ty) - Ix*Iz*tdoty*tdotz + Iy*tauz*cos(tx)*sin(ty) + Iz*tauy*sin(tx)*sin(ty) - Iy**2*tdotx*tdoty*cos(tx)**2*sin(ty) + Iz**2*tdotx*tdoty*cos(tx)**2*sin(ty) - Ix*Iy*tdoty*tdotz*cos(tx)**2 + Ix*Iz*tdoty*tdotz*cos(tx)**2 + Ix*Iz*tdotx*tdoty*sin(ty) + Iy**2*tdotz**2*cos(tx)*cos(ty)**3*sin(tx) - Iz**2*tdotz**2*cos(tx)*cos(ty)**3*sin(tx) - Ix*Iy*tdotz**2*cos(tx)*cos(ty)**3*sin(tx) + Ix*Iz*tdotz**2*cos(tx)*cos(ty)**3*sin(tx) + Ix*Iy*tdotx*tdoty*cos(tx)**2*sin(ty) - Ix*Iz*tdotx*tdoty*cos(tx)**2*sin(ty)))/(Iy*Iz*cos(ty)**2)
        theta_A[2, 2] = 1
        theta_A[2, 3] = -(dt**2*(Iy**2*tdoty*cos(tx)**2 + Iz**2*tdoty*sin(tx)**2 - Ix*Iy*tdoty*cos(tx)**2 - Ix*Iz*tdoty*sin(tx)**2 + Iy**2*tdotz*cos(tx)*cos(ty)*sin(tx) - Iz**2*tdotz*cos(tx)*cos(ty)*sin(tx) - Ix*Iy*tdotz*cos(tx)*cos(ty)*sin(tx) + Ix*Iz*tdotz*cos(tx)*cos(ty)*sin(tx)))/(Iy*Iz*cos(ty))
        theta_A[2, 4] = -(dt**2*(tdotx - tdotz*sin(ty))*(Iy**2*cos(tx)**2 + Iz**2*sin(tx)**2 - Ix*Iy*cos(tx)**2 - Ix*Iz*sin(tx)**2))/(Iy*Iz*cos(ty))
        theta_A[2, 5] = (dt*(Iy*Iz*cos(tx)**2*cos(ty) + Iy*Iz*cos(ty)*sin(tx)**2 + Iy**2*dt*tdoty*cos(tx)**2*sin(ty) + Iz**2*dt*tdoty*sin(tx)**2*sin(ty) - Iy**2*dt*tdotx*cos(tx)*cos(ty)*sin(tx) + Iz**2*dt*tdotx*cos(tx)*cos(ty)*sin(tx) - Ix*Iy*dt*tdoty*cos(tx)**2*sin(ty) - Ix*Iz*dt*tdoty*sin(tx)**2*sin(ty) + 2*Iy**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) - 2*Iz**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) + Ix*Iy*dt*tdotx*cos(tx)*cos(ty)*sin(tx) - Ix*Iz*dt*tdotx*cos(tx)*cos(ty)*sin(tx) - 2*Ix*Iy*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) + 2*Ix*Iz*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty)))/(Iy*Iz*cos(ty))
        theta_A[3, 0] = (Iy*Iz**2*dt*tdotz**2*cos(ty)**3*sin(tx)**2 - Iy**2*Iz*dt*tdotz**2*cos(ty)**3*sin(tx)**2 + Ix*Iz*dt*tauy*cos(tx)*sin(ty) + Iy*Iz**2*dt*tdoty**2*cos(tx)**2*cos(ty) - Iy**2*Iz*dt*tdoty**2*cos(tx)**2*cos(ty) - Ix*Iy*dt*tauz*sin(tx)*sin(ty) - Iy*Iz**2*dt*tdoty**2*cos(ty)*sin(tx)**2 + Iy**2*Iz*dt*tdoty**2*cos(ty)*sin(tx)**2 - Iy*Iz**2*dt*tdotz**2*cos(tx)**2*cos(ty)**3 + Iy**2*Iz*dt*tdotz**2*cos(tx)**2*cos(ty)**3 + Ix*Iy**2*dt*tdotz**2*cos(tx)**2*cos(ty)*sin(ty)**2 - Ix**2*Iy*dt*tdotz**2*cos(tx)**2*cos(ty)*sin(ty)**2 - Ix*Iz**2*dt*tdotz**2*cos(tx)**2*cos(ty)*sin(ty)**2 + Ix**2*Iz*dt*tdotz**2*cos(tx)**2*cos(ty)*sin(ty)**2 - Ix*Iy**2*dt*tdotz**2*cos(ty)*sin(tx)**2*sin(ty)**2 + Ix**2*Iy*dt*tdotz**2*cos(ty)*sin(tx)**2*sin(ty)**2 + Ix*Iz**2*dt*tdotz**2*cos(ty)*sin(tx)**2*sin(ty)**2 - Ix**2*Iz*dt*tdotz**2*cos(ty)*sin(tx)**2*sin(ty)**2 + 2*Ix*Iy**2*dt*tdotx*tdoty*cos(tx)*sin(tx)*sin(ty) - 2*Ix**2*Iy*dt*tdotx*tdoty*cos(tx)*sin(tx)*sin(ty) - 2*Ix*Iz**2*dt*tdotx*tdoty*cos(tx)*sin(tx)*sin(ty) + 2*Ix**2*Iz*dt*tdotx*tdoty*cos(tx)*sin(tx)*sin(ty) - Ix*Iy**2*dt*tdotx*tdotz*cos(tx)**2*cos(ty)*sin(ty) + Ix**2*Iy*dt*tdotx*tdotz*cos(tx)**2*cos(ty)*sin(ty) + Ix*Iz**2*dt*tdotx*tdotz*cos(tx)**2*cos(ty)*sin(ty) - Ix**2*Iz*dt*tdotx*tdotz*cos(tx)**2*cos(ty)*sin(ty) + 4*Iy*Iz**2*dt*tdoty*tdotz*cos(tx)*cos(ty)**2*sin(tx) - 4*Iy**2*Iz*dt*tdoty*tdotz*cos(tx)*cos(ty)**2*sin(tx) - 2*Ix*Iy**2*dt*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty)**2 + 2*Ix**2*Iy*dt*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty)**2 + Ix*Iy**2*dt*tdotx*tdotz*cos(ty)*sin(tx)**2*sin(ty) - Ix**2*Iy*dt*tdotx*tdotz*cos(ty)*sin(tx)**2*sin(ty) + 2*Ix*Iz**2*dt*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty)**2 - 2*Ix**2*Iz*dt*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty)**2 - Ix*Iz**2*dt*tdotx*tdotz*cos(ty)*sin(tx)**2*sin(ty) + Ix**2*Iz*dt*tdotx*tdotz*cos(ty)*sin(tx)**2*sin(ty))/(Ix*Iy*Iz*cos(ty))
        theta_A[3, 1] = (Ix**2*Iz*dt*tdotx*tdoty - Ix*Iz**2*dt*tdotx*tdoty + Ix*Iy*dt*tauz*cos(tx) + Ix*Iz*dt*tauy*sin(tx) + Ix*Iz**2*dt*tdoty*tdotz*sin(ty) - Ix**2*Iz*dt*tdoty*tdotz*sin(ty) - Ix*Iy**2*dt*tdotx*tdoty*cos(tx)**2 + Ix**2*Iy*dt*tdotx*tdoty*cos(tx)**2 + Ix*Iz**2*dt*tdotx*tdoty*cos(tx)**2 - Ix**2*Iz*dt*tdotx*tdoty*cos(tx)**2 + Ix*Iy**2*dt*tdoty*tdotz*cos(tx)**2*sin(ty) - Ix**2*Iy*dt*tdoty*tdotz*cos(tx)**2*sin(ty) - Ix*Iz**2*dt*tdoty*tdotz*cos(tx)**2*sin(ty) + Ix**2*Iz*dt*tdoty*tdotz*cos(tx)**2*sin(ty) + Ix*Iz**2*dt*tdoty*tdotz*cos(ty)**2*sin(ty) - Ix**2*Iz*dt*tdoty*tdotz*cos(ty)**2*sin(ty) - Iy*Iz**2*dt*tdoty*tdotz*cos(ty)**2*sin(ty) + Iy**2*Iz*dt*tdoty*tdotz*cos(ty)**2*sin(ty) + 2*Ix*Iy**2*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) - 2*Ix**2*Iy*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) - 2*Ix*Iz**2*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) + 2*Ix**2*Iz*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) + 2*Iy*Iz**2*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) - 2*Iy**2*Iz*dt*tdotz**2*cos(tx)*cos(ty)**3*sin(tx)*sin(ty) - Ix*Iy**2*dt*tdotx*tdotz*cos(tx)*cos(ty)**3*sin(tx) + Ix**2*Iy*dt*tdotx*tdotz*cos(tx)*cos(ty)**3*sin(tx) + Ix*Iz**2*dt*tdotx*tdotz*cos(tx)*cos(ty)**3*sin(tx) - Ix**2*Iz*dt*tdotx*tdotz*cos(tx)*cos(ty)**3*sin(tx) + Ix*Iy**2*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty) - Ix**2*Iy*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty) - Ix*Iz**2*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty) + Ix**2*Iz*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty) + 2*Iy*Iz**2*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty) - 2*Iy**2*Iz*dt*tdoty*tdotz*cos(tx)**2*cos(ty)**2*sin(ty))/(Ix*Iy*Iz*cos(ty)**2)
        theta_A[3, 3] = (Iy*Iz*cos(ty) - Iy**2*dt*tdoty*cos(tx)**2*sin(ty) - Iz**2*dt*tdoty*sin(tx)**2*sin(ty) + Ix*Iy*dt*tdoty*cos(tx)**2*sin(ty) + Ix*Iz*dt*tdoty*sin(tx)**2*sin(ty) - Iy**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) + Iz**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) + Ix*Iy*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) - Ix*Iz*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty))/(Iy*Iz*cos(ty))
        theta_A[3, 4] = -(dt*(Iy*Iz**2*tdotz*cos(tx)**2*cos(ty)**2 - Iy**2*Iz*tdotz*cos(tx)**2*cos(ty)**2 - Ix*Iy**2*tdotz*cos(tx)**2*sin(ty)**2 + Ix**2*Iy*tdotz*cos(tx)**2*sin(ty)**2 - Iy*Iz**2*tdotz*cos(ty)**2*sin(tx)**2 + Iy**2*Iz*tdotz*cos(ty)**2*sin(tx)**2 - Ix*Iz**2*tdotz*sin(tx)**2*sin(ty)**2 + Ix**2*Iz*tdotz*sin(tx)**2*sin(ty)**2 + Ix*Iy**2*tdotx*cos(tx)**2*sin(ty) - Ix**2*Iy*tdotx*cos(tx)**2*sin(ty) + Ix*Iz**2*tdotx*sin(tx)**2*sin(ty) - Ix**2*Iz*tdotx*sin(tx)**2*sin(ty) - 2*Iy*Iz**2*tdoty*cos(tx)*cos(ty)*sin(tx) + 2*Iy**2*Iz*tdoty*cos(tx)*cos(ty)*sin(tx)))/(Ix*Iy*Iz*cos(ty))
        theta_A[3, 5] = (Ix*Iy*Iz*cos(tx)**2*cos(ty)*sin(ty) - (Ix*Iy*Iz*sin(2*ty))/2 + Ix*Iy*Iz*cos(ty)*sin(tx)**2*sin(ty) - Iy*Iz**2*dt*tdoty*cos(tx)**2*cos(ty)**2 + Iy**2*Iz*dt*tdoty*cos(tx)**2*cos(ty)**2 + Ix*Iy**2*dt*tdoty*cos(tx)**2*sin(ty)**2 - Ix**2*Iy*dt*tdoty*cos(tx)**2*sin(ty)**2 + Iy*Iz**2*dt*tdoty*cos(ty)**2*sin(tx)**2 - Iy**2*Iz*dt*tdoty*cos(ty)**2*sin(tx)**2 + Ix*Iz**2*dt*tdoty*sin(tx)**2*sin(ty)**2 - Ix**2*Iz*dt*tdoty*sin(tx)**2*sin(ty)**2 - 2*Iy*Iz**2*dt*tdotz*cos(tx)*cos(ty)**3*sin(tx) + 2*Iy**2*Iz*dt*tdotz*cos(tx)*cos(ty)**3*sin(tx) + 2*Ix*Iy**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty)**2 - 2*Ix**2*Iy*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty)**2 - 2*Ix*Iz**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty)**2 + 2*Ix**2*Iz*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty)**2 - Ix*Iy**2*dt*tdotx*cos(tx)*cos(ty)*sin(tx)*sin(ty) + Ix**2*Iy*dt*tdotx*cos(tx)*cos(ty)*sin(tx)*sin(ty) + Ix*Iz**2*dt*tdotx*cos(tx)*cos(ty)*sin(tx)*sin(ty) - Ix**2*Iz*dt*tdotx*cos(tx)*cos(ty)*sin(tx)*sin(ty))/(Ix*Iy*Iz*cos(ty))
        theta_A[4, 0] = -(dt*(Iz*tauy*sin(tx) + Iy**2*tdotx*tdoty - Iz**2*tdotx*tdoty + Iy*tauz*cos(tx) - 2*Iy**2*tdotx*tdoty*cos(tx)**2 + 2*Iz**2*tdotx*tdoty*cos(tx)**2 - Iy**2*tdoty*tdotz*sin(ty) + Iz**2*tdoty*tdotz*sin(ty) - Ix*Iy*tdotx*tdoty + Ix*Iz*tdotx*tdoty + 2*Iy**2*tdoty*tdotz*cos(tx)**2*sin(ty) - 2*Iz**2*tdoty*tdotz*cos(tx)**2*sin(ty) + 2*Ix*Iy*tdotx*tdoty*cos(tx)**2 - 2*Ix*Iz*tdotx*tdoty*cos(tx)**2 + Ix*Iy*tdoty*tdotz*sin(ty) - Ix*Iz*tdoty*tdotz*sin(ty) + 2*Iy**2*tdotz**2*cos(tx)*cos(ty)*sin(tx)*sin(ty) - 2*Iz**2*tdotz**2*cos(tx)*cos(ty)*sin(tx)*sin(ty) - 2*Iy**2*tdotx*tdotz*cos(tx)*cos(ty)*sin(tx) + 2*Iz**2*tdotx*tdotz*cos(tx)*cos(ty)*sin(tx) - 2*Ix*Iy*tdoty*tdotz*cos(tx)**2*sin(ty) + 2*Ix*Iz*tdoty*tdotz*cos(tx)**2*sin(ty) - 2*Ix*Iy*tdotz**2*cos(tx)*cos(ty)*sin(tx)*sin(ty) + 2*Ix*Iz*tdotz**2*cos(tx)*cos(ty)*sin(tx)*sin(ty) + 2*Ix*Iy*tdotx*tdotz*cos(tx)*cos(ty)*sin(tx) - 2*Ix*Iz*tdotx*tdotz*cos(tx)*cos(ty)*sin(tx)))/(Iy*Iz)
        theta_A[4, 1] = (dt*tdotz*(Iy**2*tdotz*sin(tx)**2*sin(ty)**2 - Iz**2*tdotx*cos(tx)**2*sin(ty) - Iy**2*tdotx*sin(tx)**2*sin(ty) - Iz**2*tdotz*cos(tx)**2*cos(ty)**2 - Iy**2*tdotz*cos(ty)**2*sin(tx)**2 + Iz**2*tdotz*cos(tx)**2*sin(ty)**2 - Iy**2*tdoty*cos(tx)*cos(ty)*sin(tx) + Iz**2*tdoty*cos(tx)*cos(ty)*sin(tx) + Ix*Iz*tdotx*cos(tx)**2*sin(ty) + Ix*Iy*tdotx*sin(tx)**2*sin(ty) + Ix*Iz*tdotz*cos(tx)**2*cos(ty)**2 + Ix*Iy*tdotz*cos(ty)**2*sin(tx)**2 - Ix*Iz*tdotz*cos(tx)**2*sin(ty)**2 - Ix*Iy*tdotz*sin(tx)**2*sin(ty)**2 + Ix*Iy*tdoty*cos(tx)*cos(ty)*sin(tx) - Ix*Iz*tdoty*cos(tx)*cos(ty)*sin(tx)))/(Iy*Iz)
        theta_A[4, 3] = (dt*((Iy**2*tdoty*sin(2*tx))/2 - (Iz**2*tdoty*sin(2*tx))/2 + Iz**2*tdotz*cos(tx)**2*cos(ty) + Iy**2*tdotz*cos(ty)*sin(tx)**2 - (Ix*Iy*tdoty*sin(2*tx))/2 + (Ix*Iz*tdoty*sin(2*tx))/2 - Ix*Iz*tdotz*cos(tx)**2*cos(ty) - Ix*Iy*tdotz*cos(ty)*sin(tx)**2))/(Iy*Iz)
        theta_A[4, 4] = (2*Iy*Iz + Iy**2*dt*tdotx*sin(2*tx) - Iz**2*dt*tdotx*sin(2*tx) - Ix*Iy*dt*tdotx*sin(2*tx) + Ix*Iz*dt*tdotx*sin(2*tx) - 2*Iy**2*dt*tdotz*cos(tx)*sin(tx)*sin(ty) + 2*Iz**2*dt*tdotz*cos(tx)*sin(tx)*sin(ty) + 2*Ix*Iy*dt*tdotz*cos(tx)*sin(tx)*sin(ty) - 2*Ix*Iz*dt*tdotz*cos(tx)*sin(tx)*sin(ty))/(2*Iy*Iz)
        theta_A[4, 5] = (dt*(Iz**2*tdotx*cos(tx)**2*cos(ty) + Iy**2*tdotx*cos(ty)*sin(tx)**2 - Iy**2*tdoty*cos(tx)*sin(tx)*sin(ty) + Iz**2*tdoty*cos(tx)*sin(tx)*sin(ty) - 2*Iz**2*tdotz*cos(tx)**2*cos(ty)*sin(ty) - Ix*Iz*tdotx*cos(tx)**2*cos(ty) - 2*Iy**2*tdotz*cos(ty)*sin(tx)**2*sin(ty) - Ix*Iy*tdotx*cos(ty)*sin(tx)**2 + Ix*Iy*tdoty*cos(tx)*sin(tx)*sin(ty) - Ix*Iz*tdoty*cos(tx)*sin(tx)*sin(ty) + 2*Ix*Iz*tdotz*cos(tx)**2*cos(ty)*sin(ty) + 2*Ix*Iy*tdotz*cos(ty)*sin(tx)**2*sin(ty)))/(Iy*Iz)
        theta_A[5, 0] = -(dt*(Iy*tauz*sin(tx) + (Iy**2*tdotz**2*sin(2*ty))/2 - (Iz**2*tdotz**2*sin(2*ty))/2 - Iz*tauy*cos(tx) - (Ix*Iy*tdotz**2*sin(2*ty))/2 + (Ix*Iz*tdotz**2*sin(2*ty))/2 - Iy**2*tdotx*tdoty*sin(2*tx) + Iz**2*tdotx*tdoty*sin(2*tx) - Iy**2*tdotx*tdotz*cos(ty) + Iz**2*tdotx*tdotz*cos(ty) + 2*Iy**2*tdotx*tdotz*cos(tx)**2*cos(ty) - 2*Iz**2*tdotx*tdotz*cos(tx)**2*cos(ty) + Ix*Iy*tdotx*tdoty*sin(2*tx) - Ix*Iz*tdotx*tdoty*sin(2*tx) + Ix*Iy*tdotx*tdotz*cos(ty) - Ix*Iz*tdotx*tdotz*cos(ty) - 2*Iy**2*tdotz**2*cos(tx)**2*cos(ty)*sin(ty) + 2*Iz**2*tdotz**2*cos(tx)**2*cos(ty)*sin(ty) + 2*Iy**2*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty) - 2*Iz**2*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty) + 2*Ix*Iy*tdotz**2*cos(tx)**2*cos(ty)*sin(ty) - 2*Ix*Iz*tdotz**2*cos(tx)**2*cos(ty)*sin(ty) - 2*Ix*Iy*tdotx*tdotz*cos(tx)**2*cos(ty) + 2*Ix*Iz*tdotx*tdotz*cos(tx)**2*cos(ty) - 2*Ix*Iy*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty) + 2*Ix*Iz*tdoty*tdotz*cos(tx)*sin(tx)*sin(ty)))/(Iy*Iz*cos(ty))
        theta_A[5, 1] = (dt*(Iz**2*tdoty*tdotz + Iy**2*tdoty*tdotz*cos(tx)**2 - Iz**2*tdoty*tdotz*cos(tx)**2 - Iz**2*tdotx*tdoty*sin(ty) - Ix*Iz*tdoty*tdotz + Iy*tauz*cos(tx)*sin(ty) + Iz*tauy*sin(tx)*sin(ty) - Iy**2*tdotx*tdoty*cos(tx)**2*sin(ty) + Iz**2*tdotx*tdoty*cos(tx)**2*sin(ty) - Ix*Iy*tdoty*tdotz*cos(tx)**2 + Ix*Iz*tdoty*tdotz*cos(tx)**2 + Ix*Iz*tdotx*tdoty*sin(ty) + Iy**2*tdotz**2*cos(tx)*cos(ty)**3*sin(tx) - Iz**2*tdotz**2*cos(tx)*cos(ty)**3*sin(tx) - Ix*Iy*tdotz**2*cos(tx)*cos(ty)**3*sin(tx) + Ix*Iz*tdotz**2*cos(tx)*cos(ty)**3*sin(tx) + Ix*Iy*tdotx*tdoty*cos(tx)**2*sin(ty) - Ix*Iz*tdotx*tdoty*cos(tx)**2*sin(ty)))/(Iy*Iz*cos(ty)**2)
        theta_A[5, 3] = -(dt*(Iy**2*tdoty*cos(tx)**2 + Iz**2*tdoty*sin(tx)**2 - Ix*Iy*tdoty*cos(tx)**2 - Ix*Iz*tdoty*sin(tx)**2 + Iy**2*tdotz*cos(tx)*cos(ty)*sin(tx) - Iz**2*tdotz*cos(tx)*cos(ty)*sin(tx) - Ix*Iy*tdotz*cos(tx)*cos(ty)*sin(tx) + Ix*Iz*tdotz*cos(tx)*cos(ty)*sin(tx)))/(Iy*Iz*cos(ty))
        theta_A[5, 4] = -(dt*(tdotx - tdotz*sin(ty))*(Iy**2*cos(tx)**2 + Iz**2*sin(tx)**2 - Ix*Iy*cos(tx)**2 - Ix*Iz*sin(tx)**2))/(Iy*Iz*cos(ty))
        theta_A[5, 5] = (Iy*Iz*cos(tx)**2*cos(ty) + Iy*Iz*cos(ty)*sin(tx)**2 + Iy**2*dt*tdoty*cos(tx)**2*sin(ty) + Iz**2*dt*tdoty*sin(tx)**2*sin(ty) - Iy**2*dt*tdotx*cos(tx)*cos(ty)*sin(tx) + Iz**2*dt*tdotx*cos(tx)*cos(ty)*sin(tx) - Ix*Iy*dt*tdoty*cos(tx)**2*sin(ty) - Ix*Iz*dt*tdoty*sin(tx)**2*sin(ty) + 2*Iy**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) - 2*Iz**2*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) + Ix*Iy*dt*tdotx*cos(tx)*cos(ty)*sin(tx) - Ix*Iz*dt*tdotx*cos(tx)*cos(ty)*sin(tx) - 2*Ix*Iy*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty) + 2*Ix*Iz*dt*tdotz*cos(tx)*cos(ty)*sin(tx)*sin(ty))/(Iy*Iz*cos(ty))
        theta_thetadot_next = theta_A @ theta_thetadot
        
        pdot_next[0] += Tau*(dt*sin(ty)*sin(tz))/m
        pdot_next[1] -= Tau*(dt*cos(tz)*sin(ty))/m
        pdot_next[2] += Tau*(dt*cos(ty))/m

        theta_B[0, 0] = dt**2/Ix
        theta_B[0, 1] = (dt**2*sin(tx)*sin(ty))/(Iy*cos(ty))
        theta_B[0, 2] = (dt**2*cos(tx)*sin(ty))/(Iz*cos(ty))
        theta_B[1, 1] = (dt**2*cos(tx))/Iy
        theta_B[1, 2] = -(dt**2*sin(tx))/Iz
        theta_B[2, 1] = (dt**2*sin(tx))/(Iy*cos(ty))
        theta_B[2, 2] = (dt**2*cos(tx))/(Iz*cos(ty))
        theta_B[3, 0] = dt/Ix
        theta_B[3, 1] = (dt*sin(tx)*sin(ty))/(Iy*cos(ty))
        theta_B[3, 2] = (dt*cos(tx)*sin(ty))/(Iz*cos(ty))
        theta_B[4, 1] = (dt*cos(tx))/Iy
        theta_B[4, 2] = -(dt*sin(tx))/Iz
        theta_B[5, 1] = (dt*sin(tx))/(Iy*cos(ty))
        theta_B[5, 2] = (dt*cos(tx))/(Iz*cos(ty))
        theta_thetadot_next += theta_B @ u[1:]

        x_next = np.concatenate((p_next, pdot_next, theta_thetadot_next)).reshape(-1)
        return x_next

    def forward_simulate(self, x, u):
        x = x.reshape(self.Dx, -1); u = u.reshape(self.Du, -1)
        x_next = np.zeros_like(x)
        px, py, pz = x[0], x[1], x[2]
        pdotx, pdoty, pdotz = x[3], x[4], x[5]
        tx, ty, tz = x[6], x[7], x[8]
        tdotx, tdoty, tdotz = x[9], x[10], x[11]
        Tau, taux, tauy, tauz = u[0], u[1], u[2], u[3]
        self.lib.compute_next_state(
            x_next.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            px.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            py.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pdotx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pdoty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pdotz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tdotx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tdoty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tdotz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Tau.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            taux.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tauy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tauz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            1, self.I[0,0], self.I[1,1], self.I[2,2], 
            self.kd, self.m, self.dt
        )
        return x_next.reshape(-1)

    # def forward_simulate(self, x, u, u_offset = None):
    #     u = np.clip(u, 0, 2)
    #     if u_offset is None:
    #         u_mag = np.sqrt(9.81/4)
    #         u_offset = np.array([u_mag]*self.Du)**2 
    #     u_act = u_offset + u**2
    #     p, pdot, theta, thetadot = x[:3], x[3:6], x[6:9], x[9:]

    #     #step
    #     omega = self.thetadot2omega(thetadot, theta)
    
    #     a = self.acceleration(u_act, theta, pdot)
    #     omegadot = self.angular_acceleration(u_act, omega)
    #     omega = omega + self.dt*omegadot
    #     thetadot= self.omega2thetadot(omega, theta)
    #     theta = theta + self.dt*thetadot
    #     pdot = pdot + self.dt*a
    #     p = p + self.dt*pdot
        
    #     x_next = np.concatenate([p, pdot, theta, thetadot])
    #     return x_next



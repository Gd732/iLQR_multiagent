# %%
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt


from arguments import add_arguments
from ilqr.multiLQR import *
from visualize import draw_car
from collections import deque
from ilqr.vehicle_model import *
# %%
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        epochs = kwargs.get('epochs', 'unknown number of')
        print(f"Running {func.__name__} for {epochs} epochs took {end_time - start_time} seconds.")
        print(f"The average time for each epoch is {(end_time - start_time)/epochs} seconds.")
        return result
    return wrapper

class SimParams:    
    agent1_start_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    agent2_start_state = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    agent3_start_state = np.array([2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# %%
def create_global_plan_1():
    x_coords = np.zeros(50)
    y_coords = np.zeros(50)
    z_coords = np.arange(0, 50)

    points = np.array([x_coords, y_coords, z_coords]).T
    return points

def create_global_plan_2():
    x_coords = np.ones(50)
    y_coords = np.ones(50)
    z_coords = np.arange(0, 50)

    points = np.array([x_coords, y_coords, z_coords]).T
    return points

def create_global_plan_3():
    y1 = np.arange(45, -5, -1)
    x1 = np.full_like(y1, 40) 

    x2 = np.arange(40, 90)
    y2 = np.full_like(x2, -5)

    x_coords = np.concatenate((x1, x2))
    y_coords = np.concatenate((y1, y2))

    points = np.array([x_coords, y_coords]).T
    return points

# %%
class multiAgentSimulator:
    def __init__(self, args, SimParams):
        self.args = args
        self.model = QuadcopterModel(args)
        self.simparams = SimParams
        self.navigation_agent1 = None
        self.navigation_agent2 = None
        self.navigation_agent3 = None
        
        self.nominal_control = np.zeros((self.args.num_ctrls, self.args.horizon))
        self.nominal_control[0,:] = np.linspace(self.args.acc_control_limits[1], 0, self.args.horizon)
        self.count = 0
        
        self.current_agent1_state_long = self.simparams.agent1_start_state
        self.current_agent2_state_long = self.simparams.agent2_start_state
        # self.current_agent3_state_long = self.simparams.agent3_start_state
        
        self.last_agent1_states = np.expand_dims(self.simulate_states(
            self.current_agent1_state_long, self.nominal_control), axis=0)
        self.last_agent2_states = np.expand_dims(self.simulate_states(
            self.current_agent2_state_long, self.nominal_control), axis=0)
        # self.last_agent3_states = np.expand_dims(self.simulate_states(
        #     self.current_agent3_state_long, self.nominal_control), axis=0)
        
        self.agent1_states_long = [self.current_agent1_state_long]
        self.agent2_states_long = [self.current_agent2_state_long]
        # self.agent3_states_long = [self.current_agent3_state_long]
        self.agent1_states_short = [self.current_agent1_state_long]
        self.agent2_states_short = [self.current_agent2_state_long]
        # self.agent3_states_short = [self.current_agent3_state_long]
        self.tmp_agent1_states_long = deque(maxlen=20)
        self.tmp_agent1_controls_long = deque(maxlen=20)
        self.tmp_agent2_states_long = deque(maxlen=20)
        self.tmp_agent2_controls_long = deque(maxlen=20)

        # self.tmp_agent3_states_long = deque(maxlen=20)
        self.tmp_agent1_lo_12 = deque(maxlen=20)
        self.tmp_agent2_lo_122 = deque(maxlen=20)

        self.agent1_controls_long = []
        self.agent2_controls_long = []
        # self.agent3_controls_long = []

        # self.global_plan_agent1_cross = self.create_global_plan(25, 70, -5)
        self.global_plan_agent1 = create_global_plan_1()
        self.global_plan_agent2 = create_global_plan_2()
        # self.global_plan_agent3 = create_global_plan_3()

        self.create_ilqr_agents(self.args.sim_options)        

    
    def reset(self):
        self.create_ilqr_agents(self.args.sim_options)        
        self.current_agent1_state_long = self.simparams.agent1_start_state
        self.current_agent2_state_long = self.simparams.agent2_start_state
        self.current_agent3_state_long = self.simparams.agent3_start_state

        self.agent1_states_long = [self.current_agent1_state_long]
        self.agent2_states_long = [self.current_agent2_state_long]
        self.agent3_states_long = [self.current_agent3_state_long]

        self.agent1_states_short = [self.current_agent1_state_long]
        self.agent2_states_short = [self.current_agent2_state_long]
        self.agent3_states_short = [self.current_agent3_state_long]

        self.count = 0
        
    def create_ilqr_agents(self, simu_options):
        global_plan_agent1 = None; global_plan_agent2 = None
        global_plan_agent1 = self.global_plan_agent1
        global_plan_agent2 = self.global_plan_agent2
        if self.args.number_of_agents == 3:
            global_plan_agent3 = self.global_plan_agent3
                
        self.navigation_agent1 = iLQR(self.args, self.args.car_dims, "1")
        self.navigation_agent1.set_global_plan(global_plan_agent1)
        
        self.navigation_agent2 = iLQR(self.args, self.args.car_dims, "2")
        self.navigation_agent2.set_global_plan(global_plan_agent2)
        
        if self.args.number_of_agents == 3:
            self.navigation_agent3 = iLQR(self.args, self.args.car_dims, "3")
            self.navigation_agent3.set_global_plan(global_plan_agent3)
        
    def get_agent1_states(self):
        agent1_states = self.current_agent1_state_long.copy()
        return agent1_states
    
    def get_agent2_states(self):
        agent2_states = self.current_agent2_state_long.copy()
        return agent2_states

    def get_agent3_states(self):
        agent3_states = self.current_agent3_state_long.copy()
        return agent3_states
    
    def get_npc_states(self, i):
        return self.NPC_states[:, i:i+self.args.horizon]
    
    def create_global_plan(self, desired_start_x, desired_end_x, desired_y):
        plan_ilqr = []
        for i in range(desired_start_x, desired_end_x):
            plan_ilqr.append(np.array([i, desired_y]))
        plan_ilqr = np.array(plan_ilqr)
        return plan_ilqr
        
    def run_model_simulation(self, state, control):
        # control = np.clip(control, self.args.acc_control_limits_drone[0], self.args.acc_control_limits_drone[1])
        next_state = self.model.forward_simulate(state, control)
        return next_state
    def run_model_simulation_with_disturbance(self, state, control, noise_type):
        """
        Find the next state of the vehicle given the current state and control input
        """
        next_state = self.model.forward_simulate(state, control)
        if noise_type == 'gaussian':
            noise = np.random.normal(0, self.args.state_gaussian_noise, size=self.args.num_states)
        return next_state + noise
    def simulate_states(self, init_state, controls):
        states_list = []
        states_list.append(init_state)
        controls = np.hstack((controls, np.zeros((self.args.num_ctrls, self.args.horizon))))
        for i in range(controls.shape[1]):
            next_state = self.run_model_simulation(states_list[i], controls[:, i])
            states_list.append(next_state)
        states_list = np.array(states_list).T
        return states_list
    

    def run_step_ilqr_with_br_short_hori_follow(self, current_epoch):
        assert self.navigation_agent1 != None, "Navigation Agent not initialized"
        # print('step', self.count, self.last_agent2_states.shape)
        # print(lane)
        
        curr_costs = np.zeros(self.args.number_of_agents)
        last_costs = np.zeros(self.args.number_of_agents)
        loop_count = 0
        current_agent1_state = self.current_agent1_state_long
        current_agent2_state = self.current_agent2_state_long
        if self.args.number_of_agents == 3:
            current_agent3_state = self.current_agent3_state_long
        while True:
            # lane = np.zeros_like(self.last_agent1_states)
            # lane[:,1,:] = -14
            # lane[:,0,:] = np.arange(lane.shape[2]) + self.last_agent1_states[0,0,0]

            # generate the npcs for agent1
            if self.args.number_of_agents == 3:
                npcs_for_agent1 = np.array([self.last_agent2_states, self.last_agent3_states])
            else:
                npcs_for_agent1 = np.array([self.last_agent2_states])
            # print(npcs_for_agent1.shape)
            states_agent1_hori, controls_agent1, curr_cost_agent1 = self.navigation_agent1.run_step_long(self.get_agent1_states(), npcs_for_agent1)
            current_agent1_state = self.run_model_simulation(self.current_agent1_state_long, controls_agent1[:, 0])
            # print(states_agent1_hori)

            # generate the npcs for agent2
            if self.args.number_of_agents == 3:
                npcs_for_agent2 = np.array([self.last_agent1_states, self.last_agent3_states])
            else:
                npcs_for_agent2 = np.array([self.last_agent1_states])
            states_agent2_hori, controls_agent2, curr_cost_agent2 = self.navigation_agent2.run_step_long(self.get_agent2_states(), npcs_for_agent2)
            current_agent2_state = self.run_model_simulation(self.current_agent2_state_long, controls_agent2[:, 0])

            # generate the npcs for agent3
            if self.args.number_of_agents == 3:
                npcs_for_agent3 = np.array([self.last_agent1_states, self.last_agent2_states])
                states_agent3_hori, controls_agent3, curr_cost_agent3 = self.navigation_agent3.run_step_long(self.get_agent3_states(), npcs_for_agent3)
                current_agent3_state = self.run_model_simulation(self.current_agent3_state_long, controls_agent3[:, 0])
            
            hs = self.args.horizon_short
            curr_costs = np.array([curr_cost_agent1, curr_cost_agent2])
            if self.args.number_of_agents == 3:
                curr_costs = np.array([curr_cost_agent1, curr_cost_agent2, curr_cost_agent3])
            loop_count += 1
            print(loop_count, ':', np.linalg.norm(curr_costs-last_costs, ord=2))#, end='\r')
            if np.linalg.norm(curr_costs-last_costs, ord=2) <= 5 or loop_count >= self.args.loop_tol:
                self.count += 1
                loop_count = 0

                self.current_agent1_state_long = current_agent1_state
                self.current_agent2_state_long = current_agent2_state
                
                X1, U1 = self.set_agent_derivatives_long(nav_agent=self.navigation_agent1, npcs=npcs_for_agent1)
                self.set_agent_derivatives_long(nav_agent=self.navigation_agent2, npcs=npcs_for_agent2)
                lo_x_12, lo_x_122 = self.navigation_agent1.constraints.get_obstacle_cost_derivatives_short(X1[:, 1:], np.squeeze(npcs_for_agent1[0], axis=0))
                # print(lo_x_122.shape)
                self.tmp_agent1_lo_12.append(lo_x_12[:, 0])
                self.tmp_agent2_lo_122.append(lo_x_122[:, :, 0])
                # --------------

                self.tmp_agent1_states_long.append(self.current_agent1_state_long)
                self.tmp_agent1_controls_long.append(controls_agent1[:, 0])
                self.tmp_agent2_states_long.append(self.current_agent2_state_long)
                self.tmp_agent2_controls_long.append(controls_agent2[:, 0])
                # self.agent2_states_long.append(self.current_agent2_state_long)
                # self.agent2_controls_long.append(controls_agent2[:, 0])

                if self.args.number_of_agents == 3:
                    self.current_agent3_state_long = current_agent3_state
                    self.agent3_states_long.append(self.current_agent3_state_long)
                    self.agent3_controls_long.append(controls_agent3[:, 0])
                # print("agent1_states_len=", len(self.agent1_states_long))
                # print(current_agent3_state)
                # to plan the short horizon 
                
                ### INSERT HERE
                self.short_hori_follow(current_epoch)
                self.agent1_states_long.append(list(self.tmp_agent1_states_long)[-1])
                self.agent1_controls_long.append(list(self.tmp_agent1_controls_long)[-1])
                self.agent2_states_long.append(list(self.tmp_agent2_states_long)[-1])
                self.agent2_controls_long.append(list(self.tmp_agent2_controls_long)[-1])
                # print(self.agent1_states_long)
                break
                
            else:
                last_costs = curr_costs
                self.last_agent1_states = np.expand_dims(states_agent1_hori, axis=0)
                self.last_agent2_states = np.expand_dims(states_agent2_hori, axis=0)
                if self.args.number_of_agents == 3:
                    self.last_agent3_states = np.expand_dims(states_agent3_hori, axis=0)

    def short_hori_follow(self, current_epoch, distr_type='joint_conditional', noise_type='gaussian'):
        hs = self.args.horizon_short
        if (current_epoch+1) % hs == 0 and current_epoch+1 != hs:
            self.sub_short_hori_follow(distr_type, noise_type)
        # plan for the rest
        if current_epoch+1 == self.args.epochs:
            extend_traj_agent1 = [list(self.tmp_agent1_states_long)[-1]] * hs
            extend_ctrl_agent1 = [list(self.tmp_agent1_controls_long)[-1]] * hs
            extend_traj_agent2 = [list(self.tmp_agent2_states_long)[-1]] * hs
            extend_ctrl_agent2 = [list(self.tmp_agent2_controls_long)[-1]] * hs

            self.tmp_agent1_states_long.extend(extend_traj_agent1)
            self.tmp_agent1_controls_long.extend(extend_ctrl_agent1)
            self.tmp_agent2_states_long.extend(extend_traj_agent2)
            self.tmp_agent2_controls_long.extend(extend_ctrl_agent2)

            self.sub_short_hori_follow(distr_type, noise_type)

    def sub_short_hori_follow(self, distr_type, noise_type):
        hs = self.args.horizon_short
        nom_traj1 = np.array(self.tmp_agent1_states_long).T
        nom_ctrl1 = np.array(self.tmp_agent1_controls_long).T
        nom_traj2 = np.array(self.tmp_agent2_states_long).T
        nom_ctrl2 = np.array(self.tmp_agent2_controls_long).T
        current_agent1_state_short = self.tmp_agent1_states_long[0]
        current_agent2_state_short = self.tmp_agent2_states_long[0]

        for t in range(hs):
            if distr_type == 'marginal' or distr_type == 'conditional':
                delta_us1, delta_xs1, Sigma_u_inv1, Sigma_u1, Sigma_x1 = self.navigation_agent1.compute_nom_traj_distribution(self.navigation_agent1.tmp_lx, self.navigation_agent1.tmp_lxx, 
                                        self.navigation_agent1.tmp_lu, self.navigation_agent1.tmp_luu, self.navigation_agent1.tmp_lux)  
                delta_us2, delta_xs2, Sigma_u_inv2, Sigma_u2, Sigma_x2 = self.navigation_agent2.compute_nom_traj_distribution(self.navigation_agent2.tmp_lx, self.navigation_agent2.tmp_lxx,
                                        self.navigation_agent2.tmp_lu, self.navigation_agent2.tmp_luu, self.navigation_agent2.tmp_lux)
            elif distr_type == 'joint_conditional':
                # print(self.tmp_agent2_lo_122)
                delta_us1, delta_xs1, Sigma_u_inv1, Sigma_u1, Sigma_x1 = self.navigation_agent1.compute_nom_npc_traj_distribution(self.navigation_agent1.tmp_lx, self.navigation_agent1.tmp_lxx, 
                                        self.navigation_agent1.tmp_lu, self.navigation_agent1.tmp_luu, self.navigation_agent1.tmp_lux, self.tmp_agent1_lo_12, self.tmp_agent2_lo_122)  
                delta_us2, delta_xs2, Sigma_u_inv2, Sigma_u2, Sigma_x2 = self.navigation_agent2.compute_nom_traj_distribution(self.navigation_agent2.tmp_lx, self.navigation_agent2.tmp_lxx,
                                        self.navigation_agent2.tmp_lu, self.navigation_agent2.tmp_luu, self.navigation_agent2.tmp_lux)                
            
            # print('Sigma_u1, Sigma_x1, Sigma_u2, Sigma_x2', Sigma_u1.shape, Sigma_x1.shape, Sigma_u2.shape, Sigma_x2.shape)
            # print(Sigma_u1)
            if distr_type == 'marginal':
                Delta_t1 = self.navigation_agent1.get_state_cost_marginal(Sigma_x1, nom_traj1)
                Delta_t2 = self.navigation_agent1.get_state_cost_marginal(Sigma_x2, nom_traj2)
                self.navigation_agent1.constraints.state_cost_t = Delta_t1[1:]
                self.navigation_agent2.constraints.state_cost_t = Delta_t2[1:]
                X_short1, U_short1, cost1 = self.navigation_agent1.run_step_short_marginal(current_agent1_state_short, nom_traj1, nom_ctrl1, t)
                X_short2, U_short2, cost2 = self.navigation_agent2.run_step_short_marginal(current_agent2_state_short, nom_traj2, nom_ctrl2, t)
                current_agent1_state_short = self.run_model_simulation_with_disturbance(current_agent1_state_short, U_short1[:, 0], noise_type)
                current_agent2_state_short = self.run_model_simulation_with_disturbance(current_agent2_state_short, U_short2[:, 0], noise_type)
            elif distr_type == 'conditional'or distr_type == 'joint_conditional': 
                Delta_t1 = self.navigation_agent1.get_state_cost_single_conditional(Sigma_x1, nom_traj1)
                Delta_t2 = self.navigation_agent2.get_state_cost_single_conditional(Sigma_x2, nom_traj2)
                self.navigation_agent1.constraints.state_cost_t = Delta_t1[1:]
                self.navigation_agent2.constraints.state_cost_t = Delta_t2[1:]
                X_short1, U_short1, cost1 = self.navigation_agent1.run_step_short_conditional(current_agent1_state_short, nom_traj1, nom_ctrl1, 0)
                X_short2, U_short2, cost2 = self.navigation_agent2.run_step_short_conditional(current_agent2_state_short, nom_traj2, nom_ctrl2, 0)
                current_agent1_state_short = self.run_model_simulation_with_disturbance(current_agent1_state_short, U_short1[:, 0], noise_type)
                current_agent2_state_short = self.run_model_simulation_with_disturbance(current_agent2_state_short, U_short2[:, 0], noise_type)
            
            self.agent1_states_short.append(list(X_short1.T)[0])
            self.agent2_states_short.append(list(X_short2.T)[0])
            self.navigation_agent1.total_cost += cost1
            self.navigation_agent2.total_cost += cost2
    

    def set_agent_derivatives_long(self, nav_agent, npcs):
        # get the cost derivatives
        agent_state = self.get_agent1_states()
        nav_agent.local_planner.set_agent_state(agent_state)
        ref_traj, poly_coeff_y, poly_coeff_z = nav_agent.local_planner.get_local_plan()
        X_0 = agent_state
        U = nav_agent.control_seq_long
        X = nav_agent.get_nominal_trajectory(X_0, U)
        fx = nav_agent.vehicle_model.get_A_matrix(X[:, 1:],  U[:, :], self.args.horizon)
        fu = nav_agent.vehicle_model.get_B_matrix(X[:, 1:],  U[:, :], self.args.horizon)
        l_x, l_xx, l_u, l_uu, l_ux = nav_agent.constraints.get_cost_derivatives_long(X[:, 1:], U, poly_coeff_y, poly_coeff_z, ref_traj[:, 0], npcs) 
        # print(lo_x_12.shape, lo_x_122.shape)
        # print(l_x.shape, l_xx.shape, l_u.shape, l_uu.shape, l_ux.shape)
        nav_agent.tmp_lx.append(l_x[:, 0])
        nav_agent.tmp_lxx.append(l_xx[:, :, 0])
        nav_agent.tmp_lu.append(l_u[:, 0])
        nav_agent.tmp_luu.append(l_uu[:, :, 0])
        nav_agent.tmp_lux.append(l_ux[:, :, 0])
        nav_agent.tmp_fx.append(fx[:, :, 0])
        nav_agent.tmp_fu.append(fu[:, :, 0])
        return X, U

    @timer_decorator
    def run_epochs_ilqr(self, epochs):
        self.reset()
        for i in range(epochs):
            self.run_step_ilqr_with_br_short_hori_follow(i)
            print('\n')
            print('Step:', self.count)
            # print('Step:', self.count)
            print(self.tmp_agent1_states_long)

    def compute_distance(self, agentx_states, agenty_states):
        return np.linalg.norm(np.array(agentx_states)[:,:2]-np.array(agenty_states)[:,:2], 2, axis=1)
    
    def visualize_frame(self, X, Y, YAW, ox = [], oy = []):
        colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k'];
        plt.figure(figsize=(6,6))
        plt.plot(ox, oy, "sk")
        for k in range(len(X[0])):
            plt.cla()
            plt.plot([0, 35], [-10, -10], color='black')
            plt.plot([55, 90], [-10, -10], color='black')
            plt.plot([45, 45], [10, 45], color='#ff9933', linestyle='--')
            plt.plot([0, 35], [10, 10], color='black')
            plt.plot([55, 90], [10, 10], color='black')
            plt.plot([45, 45], [-10, -45], color='#ff9933', linestyle='--')
            plt.plot([35, 35], [-10, -45], color='black')
            plt.plot([55, 55], [-10, -45], color='black')
            plt.plot([0, 35], [0, 0], color='#ff9933', linestyle='--')
            plt.plot([55, 55], [10, 45], color='black')
            plt.plot([35, 35], [10, 45], color='black')
            plt.plot([55, 90], [0, 0], color='#ff9933', linestyle='--')             
            idx_car = 0
            for x, y, yaw in zip(X, Y, YAW):
                plt.plot(x, y, linewidth=1.5, color=colors[idx_car], label='Agent '+str(idx_car+1))
                draw_car(x[-1], y[-1], yaw[-1], 'dimgray')
                draw_car(x[k], y[k], yaw[k], colors[idx_car])
                idx_car += 1
                plt.plot(self.global_plan_agent1[::4,0], 
                self.global_plan_agent1[::4,1], 
                'o', color=(1, 0, 0, 0.2), label='Global Plan Agent 1')        
                plt.plot(self.global_plan_agent2[::3,0], 
                self.global_plan_agent2[::3,1], 
                '*', color=(0, 0, 1, 0.2), label='Global Plan Agent 2')  
                if self.args.number_of_agents == 3:
                    plt.plot(self.global_plan_agent3[::3,0], 
                    self.global_plan_agent3[::3,1], 
                    '+', color=(0, 1, 0, 0.5), label='Global Plan Agent 3')  

            plt.legend()
            plt.title("iLQR Solution")
            plt.axis("equal")
            plt.pause(0.1)

        plt.show()
    
    def visualize_all(self, interval):
        if self.args.number_of_agents == 2:
            self.visualize_frame([list(np.array(self.agent1_states_long)[::interval, 0]), list(np.array(self.agent2_states_long)[::interval, 0])],
                [list(np.array(self.agent1_states_long)[::interval, 1]), list(np.array(self.agent2_states_long)[::interval, 1])], 
                [list(np.array(self.agent1_states_long)[::interval, 3]), list(np.array(self.agent2_states_long)[::interval, 3])])
        elif self.args.number_of_agents == 3:
            self.visualize_frame([list(np.array(self.agent1_states_long)[::interval, 0]), list(np.array(self.agent2_states_long)[::interval, 0]), list(np.array(self.agent3_states_long)[::interval, 0])],
                [list(np.array(self.agent1_states_long)[::interval, 1]), list(np.array(self.agent2_states_long)[::interval, 1]), list(np.array(self.agent3_states_long)[::interval, 1])], 
                [list(np.array(self.agent1_states_long)[::interval, 3]), list(np.array(self.agent2_states_long)[::interval, 3]), list(np.array(self.agent3_states_long)[::interval, 3])])

    def visualize_all_short(self, interval):
        if self.args.number_of_agents == 2:
            self.visualize_frame([list(np.array(self.agent1_states_short)[::interval, 0]), list(np.array(self.agent2_states_short)[::interval, 0])],
                [list(np.array(self.agent1_states_short)[::interval, 1]), list(np.array(self.agent2_states_short)[::interval, 1])], 
                [list(np.array(self.agent1_states_short)[::interval, 3]), list(np.array(self.agent2_states_short)[::interval, 3])])
        elif self.args.number_of_agents == 3:
            self.visualize_frame([list(np.array(self.agent1_states_short)[::interval, 0]), list(np.array(self.agent2_states_short)[::interval, 0]), list(np.array(self.agent3_states_long)[::interval, 0])],
                [list(np.array(self.agent1_states_short)[::interval, 1]), list(np.array(self.agent2_states_short)[::interval, 1]), list(np.array(self.agent3_states_long)[::interval, 1])], 
                [list(np.array(self.agent1_states_short)[::interval, 3]), list(np.array(self.agent2_states_short)[::interval, 3]), list(np.array(self.agent3_states_long)[::interval, 3])])
# %% [markdown]
# ---

# %%
if __name__ == "__main__":
    argparser = argparse.ArgumentParser([])
    add_arguments(argparser)
    argparser.add_argument('--number_of_agents', type=int, default=2, help='Number of agents')
    argparser.add_argument('--sim_options', default="cross", type=str, help="Type of simulation, cross")
    argparser.add_argument('--epochs', type=int, default=30, help='Total number of epochs for all agents')
    argparser.add_argument('--draw_interval', type=int, default=2, help='Interval for drawing the simulation')
    argparser.add_argument('--loop_tol', type=int, default=5, help='tolerance of the loop')
    argparser.add_argument('--state_gaussian_noise', nargs="*", type=float, default=[0.05, 0.05, 0.05, 0.05], help='State disturbance: x, y, v, yaw')
    args = argparser.parse_args([])

    masim = multiAgentSimulator(args, SimParams)
    masim.run_epochs_ilqr(epochs=args.epochs)
    # print(len(masim.agent1_states_long))
    # print(len(masim.agent1_states_short))
    # print(len(masim.agent2_states_long))
    # print(len(masim.agent2_states_short))
    # print("cost_agent1=", masim.navigation_agent1.total_cost)   
    # print("cost_agent2=", masim.navigation_agent2.total_cost)
    # masim.visualize_all_short(interval=args.draw_interval)

    # %%
    # masim.visualize_all(1)

    # # %%
    # plt.plot(np.array(masim.agent1_states)[:,2], label='agent1')
    # plt.plot(np.array(masim.agent2_states)[:,2], label='agent2')
    # plt.plot(np.array(masim.agent3_states)[:,2], label='agent3')
    # plt.legend()
    # plt.show()

    # # %%
    # plt.plot(np.array(masim.agent1_controls)[:,0], label='agent1')
    # plt.plot(np.array(masim.agent2_controls)[:,0], label='agent2')
    # plt.plot(np.array(masim.agent3_controls)[:,0], label='agent3')
    # plt.legend()
    # plt.show()

    # # %%
    # plt.plot(np.linalg.norm(np.array(masim.agent1_states)[:,:2]-np.array(masim.agent2_states)[:,:2], 2, axis=1), label="12")
    # plt.plot(np.linalg.norm(np.array(masim.agent1_states)[:,:2]-np.array(masim.agent3_states)[:,:2], 2, axis=1), label="13")
    # plt.plot(np.linalg.norm(np.array(masim.agent3_states)[:,:2]-np.array(masim.agent2_states)[:,:2], 2, axis=1), label="23")
    # plt.legend()
    # plt.show()



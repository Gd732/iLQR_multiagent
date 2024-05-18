import argparse

import numpy as np
import matplotlib.pyplot as plt

from arguments import add_arguments
from ilqr.multiiLQR import multiiLQR
from visualize import draw_car


class SimParams:
    sim_options = 'follow' # 'overtake' or 'follow'
    sim_epochs = 120
    dt = 0.1
    ## Car Parameters
    car_dims = np.array([4, 2])
    agent1_start_state = np.array([5, -3, 0, 0])
    agent2_start_state = np.array([0, 2, 0, 0])
    max_speed = 180/3.6
    wheelbase = 3.0
    steer_min = -1.0
    steer_max = 1.0
    accel_min = -5.5
    accel_max = 3.0
    desired_y = 2.0
    nominal_max_acc = 0.75
    

class MultiagentSimulator:
    def __init__(self, args, SimParams):
        self.args = args

        self.simparams = SimParams
        self.navigation_agent1 = None
        self.navigation_agent2 = None

        self.nominal_control = np.zeros((2, self.args.horizon))
        self.nominal_control[0,:] = np.linspace(self.simparams.nominal_max_acc, 0, self.args.horizon)
        
        self.count = 0
        
        self.current_agent1_state = self.simparams.agent1_start_state
        self.current_agent2_state = self.simparams.agent2_start_state
        self.last_agent1_states = self.simulate_states(self.current_agent1_state, self.nominal_control)
        self.last_agent2_states = self.simulate_states(self.current_agent2_state, self.nominal_control)
        self.agent1_states = [self.current_agent1_state]
        self.agent2_states = [self.current_agent2_state]

        self.global_plan_agent1_following = self.create_global_plan(25, 70, -3)
        self.global_plan_agent2_following = self.create_global_plan(5, 50, -3)
        self.global_plan_agent1_overtaking = self.create_global_plan(5, 50, -3)
        self.global_plan_agent2_overtaking = self.create_global_plan(25, 70, -3)
        
        self.create_ilqr_agents(self.simparams.sim_options)        

        
    def reset(self):
        self.create_ilqr_agents(self.simparams.sim_options)        
        self.current_agent1_state = self.simparams.agent1_start_state
        self.current_agent2_state = self.simparams.agent2_start_state
        self.agent1_states = [self.current_agent1_state]
        self.agent2_states = [self.current_agent2_state]
        self.count = 0
        
    def create_ilqr_agents(self, simu_options):
        global_plan_agent1 = None; global_plan_agent2 = None
        if simu_options == 'follow':
            global_plan_agent1 = self.global_plan_agent1_following
            global_plan_agent2 = self.global_plan_agent2_following
        elif simu_options == 'overtake':
            global_plan_agent1 = self.global_plan_agent1_overtaking
            global_plan_agent2 = self.global_plan_agent2_overtaking
        self.navigation_agent1 = multiiLQR(self.args, self.simparams.car_dims)
        self.navigation_agent1.set_global_plan(global_plan_agent1)
        
        self.navigation_agent2 = multiiLQR(self.args, self.simparams.car_dims)
        self.navigation_agent2.set_global_plan(global_plan_agent2)
        
    def get_agent1_states(self):
        agent1_states = np.array([self.current_agent1_state[0], self.current_agent1_state[1], 
                               self.current_agent1_state[2], self.current_agent1_state[3]])
        return agent1_states
    
    def get_agent2_states(self):
        agent2_states = np.array([self.current_agent2_state[0], self.current_agent2_state[1], 
                               self.current_agent2_state[2], self.current_agent2_state[3]])
        return agent2_states

    def get_npc_states(self, i):
        return self.NPC_states[:, i:i+self.args.horizon]
    
    def create_global_plan(self, desired_start_x, desired_end_x, desired_y):
        plan_ilqr = []
        for i in range(desired_start_x, desired_end_x):
            plan_ilqr.append(np.array([i, desired_y]))
        plan_ilqr = np.array(plan_ilqr)
        return plan_ilqr
        
    def run_model_simulation(self, state, control):
        """
        Find the next state of the vehicle given the current state and control input
        """
        # Clips the controller values between min and max accel and steer values
        control[0] = np.clip(control[0], self.simparams.accel_min, self.simparams.accel_max)
        control[1] = np.clip(control[1], state[2]*np.tan(self.simparams.steer_min)/self.simparams.wheelbase, state[2]*np.tan(self.simparams.steer_max)/self.simparams.wheelbase)
        
        Ts = self.simparams.dt
        next_state = np.array([state[0] + np.cos(state[3])*(state[2]*Ts + (control[0]*Ts**2)/2),
                               state[1] + np.sin(state[3])*(state[2]*Ts + (control[0]*Ts**2)/2),
                               np.clip(state[2] + control[0]*Ts, 0.0, self.simparams.max_speed),
                              (state[3] + control[1]*Ts)%(2*np.pi)])

        return next_state
    
    def simulate_states(self, init_state, controls):
        states_list = []
        states_list.append(init_state)
        controls = np.hstack((controls, np.zeros((2, self.args.horizon))))
        for i in range(controls.shape[1]):
            next_state = self.run_model_simulation(states_list[i], controls[:, i])
            states_list.append(next_state)
        states_list = np.array(states_list).T
        return states_list
    
    def run_step_ilqr(self):
        assert self.navigation_agent1 != None, "Navigation Agent not initialized"
        
        states_agent1, controls_agent1 = self.navigation_agent1.run_step(self.get_agent1_states(), self.last_agent2_states)
        self.current_agent1_state = self.run_model_simulation(self.current_agent1_state, controls_agent1[:, 0])
        self.agent1_states.append(self.current_agent1_state)
        
        states_agent2, controls_agent2 = self.navigation_agent2.run_step(self.get_agent2_states(), self.last_agent1_states)
        self.current_agent2_state = self.run_model_simulation(self.current_agent2_state, controls_agent2[:, 0])
        self.agent2_states.append(self.current_agent2_state)
        
        self.last_agent1_states = states_agent1
        self.last_agent2_states = states_agent2
        
        self.count += 1

    def run_epochs_ilqr(self, epochs):
        self.reset()
        print('Running iLQR for', epochs, 'epochs')
        print('Starting iLQR Simulation...')
        for i in range(epochs):
            self.run_step_ilqr()
            print('Step:', self.count, end='\r')
    
    def visualize_frame(self, X, Y, YAW, ox = [], oy = []):
        colors = ['r', 'b', 'g','c', 'm', 'y', 'k'];
        for k in range(len(X[0])):
            plt.cla()
            plt.plot(ox, oy, "sk")
            
            idx_car = 0
            for x, y, yaw in zip(X, Y, YAW):
                plt.plot(x, y, linewidth=1.5, color=colors[idx_car], label='Agent '+str(idx_car))
                draw_car(x[-1], y[-1], yaw[-1], 'dimgray')
                draw_car(x[k], y[k], yaw[k], colors[idx_car])
                plt.axhline(y=-7, linestyle='-', color='k')
                plt.axhline(y=7, linestyle='-', color='k')
                plt.axhline(y=0, linestyle='--', color='k')    
                idx_car += 1
                
            plt.plot(self.global_plan_agent1_following[::4,0], 
            self.global_plan_agent1_following[::4,1], 
            'o', color=(1, 0, 0, 0.5), label='Global Plan Agent 1', markersize=4)        
            plt.plot(self.global_plan_agent2_following[::3,0], 
            self.global_plan_agent2_following[::3,1], 
            '*', color=(0, 0, 1, 0.5), label='Global Plan Agent 2', markersize=4)  
            plt.legend()
            plt.title("iLQR Solution")
            plt.axis("equal")
            plt.pause(0.1)
        plt.show()
    
    def visualize_all(self, interval):
        self.visualize_frame([list(np.array(self.agent1_states)[::interval, 0]), list(np.array(self.agent2_states)[::interval, 0])],
                [list(np.array(self.agent1_states)[::interval, 1]), list(np.array(self.agent2_states)[::interval, 1])], 
                [list(np.array(self.agent1_states)[::interval, 3]), list(np.array(self.agent2_states)[::interval, 3])])
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    add_arguments(argparser)
    args = argparser.parse_args(args=[])
    
    multisim = MultiagentSimulator(args, SimParams)
    multisim.run_epochs_ilqr(epochs=SimParams.sim_epochs)
    multisim.visualize_all(interval=3)
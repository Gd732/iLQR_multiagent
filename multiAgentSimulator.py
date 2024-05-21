import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from arguments import add_arguments
from ilqr.multiLQR import iLQR
from visualize import draw_car

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
    agent1_start_state = np.array([5, -3, 0, 0])
    agent2_start_state = np.array([0, 3, 0, 0])
    agent3_start_state = np.array([10, 3, 0, 0])

    desired_y = 2.0

class multiAgentSimulator:
    def __init__(self, args, SimParams):
        self.args = args

        self.simparams = SimParams
        self.navigation_agent1 = None
        self.navigation_agent2 = None
        self.navigation_agent3 = None
        
        self.nominal_control = np.zeros((2, self.args.horizon))
        self.nominal_control[0,:] = np.linspace(self.args.acc_control_limits[1], 0, self.args.horizon)
        
        self.count = 0
        
        self.current_agent1_state = self.simparams.agent1_start_state
        self.current_agent2_state = self.simparams.agent2_start_state
        self.current_agent3_state = self.simparams.agent3_start_state
        
        self.last_agent1_states = np.expand_dims(self.simulate_states(
            self.current_agent1_state, self.nominal_control), axis=0)
        self.last_agent2_states = np.expand_dims(self.simulate_states(
            self.current_agent2_state, self.nominal_control), axis=0)
        self.last_agent3_states = np.expand_dims(self.simulate_states(
            self.current_agent3_state, self.nominal_control), axis=0)
        
        self.agent1_states = [self.current_agent1_state]
        self.agent2_states = [self.current_agent2_state]
        self.agent3_states = [self.current_agent3_state]
        
        self.global_plan_agent1_following = self.create_global_plan(25, 70, -3)
        self.global_plan_agent2_following = self.create_global_plan(0, 50, -3)
        self.global_plan_agent3_following = self.create_global_plan(10, 60, -3)
        
        self.global_plan_agent1_overtaking = self.create_global_plan(5, 50, -3)
        self.global_plan_agent2_overtaking = self.create_global_plan(25, 60, -3)
        self.global_plan_agent3_overtaking = self.create_global_plan(15, 70, -3)

        self.create_ilqr_agents(self.args.sim_options)        

        
    def reset(self):
        self.create_ilqr_agents(self.args.sim_options)        
        self.current_agent1_state = self.simparams.agent1_start_state
        self.current_agent2_state = self.simparams.agent2_start_state
        self.current_agent3_state = self.simparams.agent3_start_state

        self.agent1_states = [self.current_agent1_state]
        self.agent2_states = [self.current_agent2_state]
        self.agent3_states = [self.current_agent3_state]

        self.count = 0
        
    def create_ilqr_agents(self, simu_options):
        global_plan_agent1 = None; global_plan_agent2 = None
        if simu_options == 'follow':
            global_plan_agent1 = self.global_plan_agent1_following
            global_plan_agent2 = self.global_plan_agent2_following
            if self.args.number_of_agents == 3:
                global_plan_agent3 = self.global_plan_agent3_following
        elif simu_options == 'overtake':
            global_plan_agent1 = self.global_plan_agent1_overtaking
            global_plan_agent2 = self.global_plan_agent2_overtaking
            if self.args.number_of_agents == 3:
                global_plan_agent3 = self.global_plan_agent3_overtaking
                
        self.navigation_agent1 = iLQR(self.args, self.args.car_dims)
        self.navigation_agent1.set_global_plan(global_plan_agent1)
        
        self.navigation_agent2 = iLQR(self.args, self.args.car_dims)
        self.navigation_agent2.set_global_plan(global_plan_agent2)
        
        if self.args.number_of_agents == 3:
            self.navigation_agent3 = iLQR(self.args, self.args.car_dims)
            self.navigation_agent3.set_global_plan(global_plan_agent3)
        
    def get_agent1_states(self):
        agent1_states = np.array([self.current_agent1_state[0], self.current_agent1_state[1], 
                               self.current_agent1_state[2], self.current_agent1_state[3]])
        return agent1_states
    
    def get_agent2_states(self):
        agent2_states = np.array([self.current_agent2_state[0], self.current_agent2_state[1], 
                               self.current_agent2_state[2], self.current_agent2_state[3]])
        return agent2_states

    def get_agent3_states(self):
        agent3_states = np.array([self.current_agent3_state[0], self.current_agent3_state[1], 
                               self.current_agent3_state[2], self.current_agent3_state[3]])
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
        """
        Find the next state of the vehicle given the current state and control input
        """
        # Clips the controller values between min and max accel and steer values
        control[0] = np.clip(control[0], self.args.acc_control_limits[0], self.args.acc_control_limits[1])
        control[1] = np.clip(control[1], state[2]*np.tan(self.args.steering_control_limits[0])/self.args.wheelbase, state[2]*np.tan(self.args.steering_control_limits[1])/self.args.wheelbase)
        
        Ts = self.args.timestep
        next_state = np.array([state[0] + np.cos(state[3])*(state[2]*Ts + (control[0]*Ts**2)/2),
                               state[1] + np.sin(state[3])*(state[2]*Ts + (control[0]*Ts**2)/2),
                               np.clip(state[2] + control[0]*Ts, 0.0, self.args.max_speed),
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
        # print('step', self.count, self.last_agent2_states.shape)
        lane = np.zeros_like(self.last_agent1_states)
        lane[:,1,:] = -14
        lane[:,0,:] = np.arange(lane.shape[2]) + self.last_agent1_states[0,0,0]
        # print(lane)
        if self.args.number_of_agents == 3:
            npcs_for_agent1 = np.array([self.last_agent2_states, self.last_agent3_states, lane])
        else:
            npcs_for_agent1 = np.array([self.last_agent2_states, lane])
        states_agent1, controls_agent1 = self.navigation_agent1.run_step(self.get_agent1_states(), npcs_for_agent1)
        self.current_agent1_state = self.run_model_simulation(self.current_agent1_state, controls_agent1[:, 0])
        self.agent1_states.append(self.current_agent1_state)
        
        if self.args.number_of_agents == 3:
            npcs_for_agent2 = np.array([self.last_agent1_states, self.last_agent3_states])
        else:
            npcs_for_agent2 = np.array([self.last_agent1_states])
        states_agent2, controls_agent2 = self.navigation_agent2.run_step(self.get_agent2_states(), npcs_for_agent2)
        self.current_agent2_state = self.run_model_simulation(self.current_agent2_state, controls_agent2[:, 0])
        self.agent2_states.append(self.current_agent2_state)
        
        if self.args.number_of_agents == 3:
            npcs_for_agent3 = np.array([self.last_agent1_states, self.last_agent2_states])
            states_agent3, controls_agent3 = self.navigation_agent3.run_step(self.get_agent3_states(), npcs_for_agent3)
            self.current_agent3_state = self.run_model_simulation(self.current_agent3_state, controls_agent3[:, 0])
            self.agent3_states.append(self.current_agent3_state)

        self.last_agent1_states = np.expand_dims(states_agent1, axis=0)
        self.last_agent2_states = np.expand_dims(states_agent2, axis=0)
        if self.args.number_of_agents == 3:
            self.last_agent3_states = np.expand_dims(states_agent3, axis=0)

        self.count += 1

    @timer_decorator
    def run_epochs_ilqr(self, epochs):
        self.reset()
        for i in range(epochs):
            self.run_step_ilqr()
            print('Step:', self.count, end='\r')
    
    def visualize_frame(self, X, Y, YAW, ox = [], oy = []):
        colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k'];
        plt.figure()
        plt.plot(ox, oy, "sk")
        for k in range(len(X[0])):
            plt.cla()
            plt.axhline(y=-7, linestyle='-', color='k')
            plt.axhline(y=7, linestyle='-', color='k')
            plt.axhline(y=0, linestyle='--', color='k')                
            idx_car = 0
            for x, y, yaw in zip(X, Y, YAW):
                plt.plot(x, y, linewidth=1.5, color=colors[idx_car], label='Agent '+str(idx_car))
                draw_car(x[-1], y[-1], yaw[-1], 'dimgray')
                draw_car(x[k], y[k], yaw[k], colors[idx_car])
                idx_car += 1
            if self.args.sim_options == 'follow':
                plt.plot(self.global_plan_agent1_following[::4,0], 
                self.global_plan_agent1_following[::4,1], 
                'o', color=(1, 0, 0, 0.2), label='Global Plan Agent 1')        
                plt.plot(self.global_plan_agent2_following[::3,0], 
                self.global_plan_agent2_following[::3,1], 
                '*', color=(0, 0, 1, 0.2), label='Global Plan Agent 2')  
                if self.args.number_of_agents == 3:
                    plt.plot(self.global_plan_agent3_following[::3,0], 
                    self.global_plan_agent3_following[::3,1], 
                    '+', color=(0, 1, 0, 0.2), label='Global Plan Agent 3')  
            elif self.args.sim_options == 'overtake':   
                plt.plot(self.global_plan_agent1_overtaking[::4,0], 
                self.global_plan_agent1_overtaking[::4,1], 
                'o', color=(1, 0, 0, 0.2), label='Global Plan Agent 1')        
                plt.plot(self.global_plan_agent2_overtaking[::3,0], 
                self.global_plan_agent2_overtaking[::3,1], 
                '*', color=(0, 0, 1, 0.2), label='Global Plan Agent 2')  
                if self.args.number_of_agents == 3:
                    plt.plot(self.global_plan_agent3_overtaking[::3,0], 
                    self.global_plan_agent3_overtaking[::3,1], 
                    '+', color=(0, 1, 0, 0.2), label='Global Plan Agent 3')

            plt.legend()
            plt.title("iLQR Solution")
            plt.axis("equal")
            plt.pause(0.1)

        plt.show()
    
    def visualize_all(self, interval):
        if self.args.number_of_agents == 2:
            self.visualize_frame([list(np.array(self.agent1_states)[::interval, 0]), list(np.array(self.agent2_states)[::interval, 0])],
                [list(np.array(self.agent1_states)[::interval, 1]), list(np.array(self.agent2_states)[::interval, 1])], 
                [list(np.array(self.agent1_states)[::interval, 3]), list(np.array(self.agent2_states)[::interval, 3])])
        elif self.args.number_of_agents == 3:
            self.visualize_frame([list(np.array(self.agent1_states)[::interval, 0]), list(np.array(self.agent2_states)[::interval, 0]), list(np.array(self.agent3_states)[::interval, 0])],
                [list(np.array(self.agent1_states)[::interval, 1]), list(np.array(self.agent2_states)[::interval, 1]), list(np.array(self.agent3_states)[::interval, 1])], 
                [list(np.array(self.agent1_states)[::interval, 3]), list(np.array(self.agent2_states)[::interval, 3]), list(np.array(self.agent3_states)[::interval, 3])])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    add_arguments(argparser)
    argparser.add_argument('--number_of_agents', type=int, default=3, help='Number of agents')
    argparser.add_argument('--sim_options', default="overtake", type=str, help="Type of simulation")
    argparser.add_argument('--epochs', type=int, default=120, help='Total number of epochs for all agents')
    argparser.add_argument('--draw_interval', type=int, default=2, help='Interval for drawing the simulation')
    args = argparser.parse_args()
    
    masim = multiAgentSimulator(args, SimParams)
    masim.run_epochs_ilqr(epochs=args.epochs)
    masim.visualize_all(interval=args.draw_interval)

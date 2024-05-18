import numpy as np
import warnings
import pdb

class LocalPlanner:
    """
    Class which creates a desired trajectory based on the global plan
    Created by iLQR class based on the plan provided by the simulator
    """
    def __init__(self, args):
        self.args = args
        self.global_plan = None
        self.agent_state = None
    
    def set_agent_state(self, agent_state):
        self.agent_state = agent_state

    def set_global_planner(self, global_plan):
        """
        Sets the global plan of the agent vehicle
        """
        self.global_plan = np.asarray(global_plan)

    def closest_node(self, node):
        closest_ind = np.sum((self.global_plan - node)**2, axis=1)
        return np.argmin(closest_ind)

    def get_local_wpts(self):
        """
        Creates a local plan based on the waypoints on the global planner 
        """
        assert self.agent_state is not None, "Ego state was not set in the LocalPlanner"

        # Find index of waypoint closest to current pose
        closest_ind = self.closest_node([self.agent_state[0],self.agent_state[1]]) 
        # local_wpts = [[global_wpts[i,0],global_wpts[i,1]] for i in range(closest_ind, closest_ind + self.args.number_of_local_wpts)]
        return self.global_plan[closest_ind:closest_ind+self.args.number_of_local_wpts] # Number of local waypoints

    
    def get_local_plan(self):
        """
        Returns the local plan based on the waypoints on the global planner
        Using a polynomial fit to smoothen the path
        """
        local_wpts = self.get_local_wpts()
        x = local_wpts[:,0]
        y = local_wpts[:,1]
        coeffs = np.polyfit(x, y, self.args.poly_order)
        new_y = np.polyval(np.poly1d(coeffs), x)

        warnings.simplefilter('ignore', np.RankWarning)
        
        return np.vstack((x, new_y)).T, coeffs

    
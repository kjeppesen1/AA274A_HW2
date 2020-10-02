import numpy as np
from numpy import linalg

V_PREV_THRES = 0.0001

class TrajectoryTracker:
    """ Trajectory tracking controller using differential flatness """
    def __init__(self, kpx, kpy, kdx, kdy, V_max=0.5, om_max=1):
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy

        self.V_max = V_max
        self.om_max = om_max

        self.coeffs = np.zeros(8) # Polynomial coefficients for x(t) and y(t) as
                                  # returned by the differential flatness code

    def reset(self):
        self.V_prev = 0
        self.om_prev = 0
        self.t_prev = 0

    def load_traj(self, times, traj):
        """ Loads in a new trajectory to follow, and resets the time """
        self.reset()
        self.traj_times = times
        self.traj = traj

    def get_desired_state(self, t):
        """
        Input:
            t: Current time
        Output:
            x_d, xd_d, xdd_d, y_d, yd_d, ydd_d: Desired state and derivatives
                at time t according to self.coeffs
        """
        x_d = np.interp(t,self.traj_times,self.traj[:,0])
        y_d = np.interp(t,self.traj_times,self.traj[:,1])
        xd_d = np.interp(t,self.traj_times,self.traj[:,3])
        yd_d = np.interp(t,self.traj_times,self.traj[:,4])
        xdd_d = np.interp(t,self.traj_times,self.traj[:,5])
        ydd_d = np.interp(t,self.traj_times,self.traj[:,6])
        
        return x_d, xd_d, xdd_d, y_d, yd_d, ydd_d

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time
        Outputs: 
            V, om: Control actions
        """

        dt = t - self.t_prev
        x_d, xd_d, xdd_d, y_d, yd_d, ydd_d = self.get_desired_state(t)

        ########## Code starts here ##########
        V = 0
        om = 0
        
        #Make sure we don't drop below V_PREV_THRESHOLD, else we may have a singularity
        if self.V_prev < V_PREV_THRES:
            self.V_prev = V_PREV_THRES
        
        #Calculate xd and yd
        xd = self.V_prev*np.cos(th)
        yd = self.V_prev*np.sin(th)
        
        #Hint, at each timestep you may consider the current velocity to be that commanded
        #in previous timestep. Controller class designed to save this as member var self.V_prev
        #use the control laws (given in HW and derived in lecture) to calculate u1, u2
        xdd = xdd_d + self.kpx*(x_d-x) + self.kdx*(xd_d-xd) #u1
        ydd = ydd_d + self.kpy*(y_d-y) + self.kdy*(yd_d-yd) #u2
        
        #Apply the equations derived in part 3i
        #Summary: Took zdd = J*[a;w] = [u1;u2] and inverted J to solve for [a;w]
        #in terms of u1 and u2 (AKA xdd and ydd)
        a = xdd*np.cos(th) + ydd*np.sin(th)
        om = -(1/self.V_prev)*(xdd*np.sin(th) - ydd*np.cos(th))
        
        #compute V by integrating a
        dt = t-self.t_prev
        V = self.V_prev + dt*a #Euler integration, where x_n+1 = x_n+dt*(x_n-x_n-1)
        
        ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        return V, om
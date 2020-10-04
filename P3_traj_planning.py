import numpy as np
from P1_astar import DetOccupancyGrid2D, AStar
from P2_rrt import *
import scipy.interpolate
import matplotlib.pyplot as plt
from HW1.P1_differential_flatness import *
from HW1.P2_pose_stabilization import *
from HW1.P3_trajectory_tracking import *

class SwitchingController(object):
    """
    Uses one controller to initially track a trajectory, then switches to a 
    second controller to regulate to the final goal.
    """
    def __init__(self, traj_controller, pose_controller, t_before_switch):
        self.traj_controller = traj_controller
        self.pose_controller = pose_controller
        self.t_before_switch = t_before_switch

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            (x,y,th): Current state 
            t: Current time

        Outputs:
            V, om: Control actions
        """
        ########## Code starts here ##########

        #grab the final times from the traj_controller object
        times = self.traj_controller.traj_times
        
        #if we are within the final time before switch, switch to the pose_controller
        if (t < times[-1] - self.t_before_switch):
            #use traj_controller
            V, om = self.traj_controller.compute_control(x, y, th, t)
        else:
            V, om = self.pose_controller.compute_control(x, y, th, t)
            
        return V, om
            
        ########## Code ends here ##########

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    
    #output a trajectory [x,y,theta,xd,yd,xdd,ydd]
    #output the times associated
    
    #convert path to numpy
    path = np.array(path)
    
    #assign times for each point in the original path (based on V_des)
    #strat is to sample dist b/w each pair of points and calc the related t given fixed V_des
    times = np.zeros(len(path))
    for i in range(1,len(path[:,0])):
        times[i] = (np.linalg.norm(path[i,:]-path[i-1,:])/V_des) + times[i-1]
        
    #now create the t_smoothed array to interpolate over
    t_smoothed = np.arange(times[0],times[-1],dt) #creates an array b/w times with interval dt
    
    #interpolate x over t
    knots_x = scipy.interpolate.splrep(times,path[:,0],s=alpha)
    
    #interpolate y over t
    knots_y = scipy.interpolate.splrep(times,path[:,1],s=alpha)
    
    #create empty array for smoothed traj, size equal to t_smoothed
    traj_smoothed = np.zeros([len(t_smoothed),7])
    
    #calc the smoothed traj, using previous results and t_smoothed
    traj_smoothed[:,0] = scipy.interpolate.splev(t_smoothed,knots_x) #x
    traj_smoothed[:,1] = scipy.interpolate.splev(t_smoothed,knots_y) #y
    traj_smoothed[:,3] = scipy.interpolate.splev(t_smoothed,knots_x,der=1) #xd
    traj_smoothed[:,4] = scipy.interpolate.splev(t_smoothed,knots_y,der=1) #yd
    traj_smoothed[:,5] = scipy.interpolate.splev(t_smoothed,knots_x,der=2) #xdd
    traj_smoothed[:,6] = scipy.interpolate.splev(t_smoothed,knots_y,der=2) #ydd
    #calc theta as the angle b/w xd and yd
    traj_smoothed[:,2] = np.arctan2(traj_smoothed[:,4],traj_smoothed[:,3]) #theta
    
    ########## Code ends here ##########

    return traj_smoothed, t_smoothed

def modify_traj_with_limits(traj, t, V_max, om_max, dt):
    """
    Modifies an existing trajectory to satisfy control limits and
    interpolates for desired timestep.

    Inputs:
        traj (np.array [N,7]): original trajecotry
        t (np.array [N]): original trajectory times
        V_max, om_max (float): control limits
        dt (float): desired timestep
    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    Hint: This should almost entirely consist of calling functions from Problem Set 1
    """
    ########## Code starts here ##########
    
    #call compute_controls from HW1
    V,om = compute_controls(traj)
    
    #rescale V to meet given control constraints
    V_tilde = rescale_V(V, om, V_max, om_max)
    
    #compute the arc length
    s = compute_arc_length(V, t)
    
    #compute new time history, tau
    tau = compute_tau(V_tilde, s)
    
    #rescale omega
    om_tilde = rescale_om(V, om, V_tilde)
    
    #interpolate the trajectory
    s_f = State(x=traj[-1,0],y=traj[-1,1],V=V_tilde[-1],th=traj[-1,2]) #populate State obect w/ final state vals
    t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)
    
    
    ########## Code ends here ##########

    return t_new, V_scaled, om_scaled, traj_scaled

import numpy as np
import matplotlib.pyplot as plt
from dubins import path_length, path_sample
from utils import plot_line_segments, line_line_intersection

# Represents a motion planning problem to be solved using the RRT algorithm
class RRTConnect(object):

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, obstacles):
        self.statespace_lo = np.array(statespace_lo)    # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = np.array(statespace_hi)    # state space upper bound (e.g., [5, 5])
        self.x_init = np.array(x_init)                  # initial state
        self.x_goal = np.array(x_goal)                  # goal state
        self.obstacles = obstacles                      # obstacle set (line segments)
        self.path = None        # the final path as a list of states

    def is_free_motion(self, obstacles, x1, x2):
        """
        Subject to the robot dynamics, returns whether a point robot moving
        along the shortest path from x1 to x2 would collide with any obstacles
        (implemented as a "black box")

        Inputs:
            obstacles: list/np.array of line segments ("walls")
            x1: start state of motion
            x2: end state of motion
        Output:
            Boolean True/False
        """
        raise NotImplementedError("is_free_motion must be overriden by a subclass of RRTConnect")

    def find_nearest_forward(self, V, x):
        """
        Given a list of states V and a query state x, returns the index (row)
        of V such that the forward steering distance (subject to robot dynamics)
        from V[i] to x is minimized

        Inputs:
            V: list/np.array of states ("samples")
            x - query state
        Output:
            Integer index of nearest point in V steering forward from x
        """
        raise NotImplementedError("find_nearest_forward must be overriden by a subclass of RRTConnect")

    def find_nearest_backward(self, V, x):
        """
        Given a list of states V and a query state x, returns the index (row)
        of V such that the forward steering distance (subject to robot dynamics)
        from x to V[i] is minimized

        Inputs:
            V: list/np.array of states ("samples")
            x - query state
        Output:
            Integer index of nearest point in V steering backward from x
        """
        raise NotImplementedError("find_nearest_backward must be overriden by a subclass of RRTConnect")

    def steer_towards_forward(self, x1, x2, eps):
        """
        Steers from x1 towards x2 along the shortest path (subject to robot
        dynamics). Returns x2 if the length of this shortest path is less than
        eps, otherwise returns the point at distance eps along the path from
        x1 to x2.

        Inputs:
            x1: start state
            x2: target state
            eps: maximum steering distance
        Output:
            State (numpy vector) resulting from bounded steering
        """
        raise NotImplementedError("steer_towards must be overriden by a subclass of RRTConnect")

    def steer_towards_backward(self, x1, x2, eps):
        """
        Steers backward from x2 towards x1 along the shortest path (subject
        to robot dynamics). Returns x1 if the length of this shortest path is
        less than eps, otherwise returns the point at distance eps along the
        path backward from x2 to x1.

        Inputs:
            x1: start state
            x2: target state
            eps: maximum steering distance
        Output:
            State (numpy vector) resulting from bounded steering
        """
        raise NotImplementedError("steer_towards_backward must be overriden by a subclass of RRTConnect")

    def solve(self, eps, max_iters = 1000):
        """
        Uses RRT-Connect to perform bidirectional RRT, with a forward tree
        rooted at self.x_init and a backward tree rooted at self.x_goal, with
        the aim of producing a dynamically-feasible and obstacle-free trajectory
        from self.x_init to self.x_goal.

        Inputs:
            eps: maximum steering distance
            max_iters: maximum number of RRT iterations (early termination
                is possible when a feasible solution is found)
                
        Output:
            None officially (just plots), but see the "Intermediate Outputs"
            descriptions below
        """
        
        state_dim = len(self.x_init)

        V_fw = np.zeros((max_iters, state_dim))     # Forward tree
        V_bw = np.zeros((max_iters, state_dim))     # Backward tree

        n_fw = 1    # the current size of the forward tree
        n_bw = 1    # the current size of the backward tree

        P_fw = -np.ones(max_iters, dtype=int)       # Stores the parent of each state in the forward tree
        P_bw = -np.ones(max_iters, dtype=int)       # Stores the parent of each state in the backward tree

        success = False

        ## Intermediate Outputs
        # You must update and/or populate:
        #    - V_fw, V_bw, P_fw, P_bw, n_fw, n_bw: the represention of the
        #           planning trees
        #    - success: whether or not you've found a solution within max_iters
        #           RRT-Connect iterations
        #    - self.path: if success is True, then must contain list of states
        #           (tree nodes) [x_init, ..., x_goal] such that the global
        #           trajectory made by linking steering trajectories connecting
        #           the states in order is obstacle-free.
        # Hint: Use your implementation of RRT as a reference

        ########## Code starts here ##########
        #init the forward and backward trees
        V_fw[0,:] = self.x_init
        V_bw[0,:] = self.x_goal
        
        #loop through
        for k in range(0,max_iters-1):
            #get random state
            x_rand = np.array([np.random.uniform(self.statespace_lo[0],self.statespace_hi[0]),
                        np.random.uniform(self.statespace_lo[1],self.statespace_hi[1])])
            #if we are doing Dubins, need 3 rand vals (theta)
            if (state_dim == 3):
                x_rand = np.append(x_rand, np.random.uniform(0,2*np.pi))
            #find the nearest neighbor forward to the random point (w/in the tree thus far, up to n)
            near_f_index = self.find_nearest_forward(V_fw[:n_fw,:], x_rand)
            x_near = V_fw[near_f_index,:]
            #find the actual new point to maybe add to the tree by scaling with eps
            x_new = self.steer_towards_forward(x_near,x_rand,eps)
            
            #check for collisions
            if self.is_free_motion(self.obstacles, x_near, x_new):
                #add the new point to the forward tree
                V_fw[n_fw,:] = x_new
                
                #add the new edge to the forward tree
                P_fw[n_fw] = near_f_index
                
                
                #find the nearest neighbor backwards
                near_b_index = self.find_nearest_backward(V_bw[:n_bw,:], x_new)
                x_connect = V_bw[near_b_index,:]
                
                #xxx
                while(True):
                    #xxxx
                    x_new_connect = self.steer_towards_backward(x_new,x_connect,eps)
                    #x_new_connect = self.steer_towards_backward(x_connect,x_new,eps)
                    
                    #check for collisons
                    if self.is_free_motion(self.obstacles, x_connect, x_new_connect):
                        #add the new point to the backward tree
                        V_bw[n_bw,:] = x_new_connect
                
                        #add the new edge to the backward tree
                        P_bw[n_bw] = near_b_index
                        
                        #n_bw = n_bw + 1
                        
                        #print("xxx")
                        #print(x_new_connect)
                        #print(x_connect)
                        #if our two points happen to be the SAME point...reconstruct the path
                        if np.linalg.norm(x_new_connect - x_new) == 0:
                            #return the completely reconstructed path
                            self.path = [x_new_connect]
                            cnt_fw = n_fw
                            cnt_bw = n_bw
                            #from connection go back to init
                            while cnt_fw > 0:
                                self.path = np.vstack([V_fw[P_fw[cnt_fw]], self.path])
                                cnt_fw = P_fw[cnt_fw]
                            #from connection go to goal
                            while cnt_bw > 0:
                                self.path = np.vstack([self.path, V_bw[P_bw[cnt_bw]]])
                                cnt_bw = P_bw[cnt_bw]
                            
                            #set success to true
                            success = True
                            break
                        n_bw = n_bw + 1
                        
                        #assign new connect to connect
                        x_connect = x_new_connect
                    else:
                        break
                        
                n_fw = n_fw + 1
                #n_bw = n_bw + 1
                        
            if(success):
                break            
            #n_fw = n_fw+1 
            #n_fw = n_fw + 1 #TESTING!
            
            ###
            #now repeat everything, but from the opposite sense
            
            #get random state
            x_rand = np.array([np.random.uniform(self.statespace_lo[0],self.statespace_hi[0]),
                        np.random.uniform(self.statespace_lo[1],self.statespace_hi[1])])
            #if we are doing Dubins, need 3 rand vals (theta)
            if (state_dim == 3):
                x_rand = np.append(x_rand, np.random.uniform(0,2*np.pi))
            #find the nearest neighbor forward to the random point (w/in the tree thus far, up to n)
            near_b_index = self.find_nearest_backward(V_bw[:n_bw,:], x_rand)
            x_near = V_bw[near_b_index,:]
            #find the actual new point to maybe add to the tree by scaling with eps
            #x_new = self.steer_towards_backward(x_near,x_rand,eps)
            x_new = self.steer_towards_backward(x_rand,x_near,eps)
            
            #check for collisions
            if self.is_free_motion(self.obstacles, x_near, x_new):
            #if self.is_free_motion(self.obstacles, x_near, x_new):
                #add the new point to the backward tree
                V_bw[n_bw,:] = x_new
                
                #add the new edge to the backward tree
                P_bw[n_bw] = near_b_index
                
                #find the nearest neighbor forward
                near_f_index = self.find_nearest_forward(V_fw[:n_fw,:], x_new)
                x_connect = V_fw[near_f_index,:]
                
                #xxx
                while(True):
                    #xxxx
                    x_new_connect = self.steer_towards_forward(x_connect,x_new,eps)
                    #x_new_connect = self.steer_towards_forward(x_new,x_connect,eps)
                    
                    #check for collisons
                    if self.is_free_motion(self.obstacles, x_new_connect, x_connect):
                        #add the new point to the forward tree
                        V_fw[n_fw,:] = x_new_connect
                
                        #add the new edge to the forward tree
                        P_fw[n_fw] = near_f_index
                        
                        #n_fw = n_fw + 1
                        
                        #print("---")
                        #print(x_new_connect)
                        #print(x_connect)
                        #if our two points happen to be the SAME point...reconstruct the path
                        if np.linalg.norm(x_new - x_new_connect) == 0:
                            #return the completely reconstructed path
                            self.path = [x_new_connect]
                            cnt_fw = n_fw
                            cnt_bw = n_bw
                            #from connection go back to init
                            while cnt_fw > 0:
                                self.path = np.vstack([V_fw[P_fw[cnt_fw]], self.path])
                                cnt_fw = P_fw[cnt_fw]
                            #from connection go to goal
                            while cnt_bw > 0:
                                self.path = np.vstack([self.path, V_bw[P_bw[cnt_bw]]])
                                cnt_bw = P_bw[cnt_bw]
                            #set success to true
                            success = True
                            break
                        
                        n_fw = n_fw + 1
                        
                        #assign new connect to connect
                        x_connect = x_new_connect
                    else:
                        break
                        
                n_bw = n_bw+1
                #n_fw = n_fw+1
            if(success):
                break
            #n_bw = n_bw+1
            #n_fw = n_fw + 1 #TESTING!

        ########## Code ends here ##########

        plt.figure()
        self.plot_problem()
        self.plot_tree(V_fw, P_fw, color="blue", linewidth=.5, label="RRTConnect forward tree")
        self.plot_tree_backward(V_bw, P_bw, color="purple", linewidth=.5, label="RRTConnect backward tree")
        
        if success:
            self.plot_path(color="green", linewidth=2, label="solution path")
            plt.scatter(V_fw[:n_fw,0], V_fw[:n_fw,1], color="blue")
            plt.scatter(V_bw[:n_bw,0], V_bw[:n_bw,1], color="purple")
        plt.scatter(V_fw[:n_fw,0], V_fw[:n_fw,1], color="blue")
        plt.scatter(V_bw[:n_bw,0], V_bw[:n_bw,1], color="purple")

        plt.show()

    def plot_problem(self):
        plot_line_segments(self.obstacles, color="red", linewidth=2, label="obstacles")
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        plt.annotate(r"$x_{init}$", self.x_init[:2] + [.2, 0], fontsize=16)
        plt.annotate(r"$x_{goal}$", self.x_goal[:2] + [.2, 0], fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

class GeometricRRTConnect(RRTConnect):
    """
    Represents a geometric planning problem, where the steering solution
    between two points is a straight line (Euclidean metric)
    """

    def find_nearest_forward(self, V, x):
        ########## Code starts here ##########
        # Hint: This should take one line.
        return np.argmin(np.sqrt((V[:,0]-x[0])**2 +(V[:,1]-x[1])**2))
        ########## Code ends here ##########

    def find_nearest_backward(self, V, x):
        return self.find_nearest_forward(V, x)

    def steer_towards_forward(self, x1, x2, eps):
        ########## Code starts here ##########
        # Hint: This should take one line.
        if (np.linalg.norm(x2-x1) < eps):
            retVal = (x2-x1) +x1
        else:
            retVal = (x2-x1)*(eps/np.linalg.norm(x2-x1)) +x1
            
        return retVal
        ########## Code ends here ##########

    def steer_towards_backward(self, x1, x2, eps):
        return self.steer_towards_forward(x2, x1, eps)

    def is_free_motion(self, obstacles, x1, x2):
        motion = np.array([x1, x2])
        for line in obstacles:
            if line_line_intersection(motion, line):
                return False
        return True

    def plot_tree(self, V, P, **kwargs):
        plot_line_segments([(V[P[i],:], V[i,:]) for i in range(V.shape[0]) if P[i] >= 0], **kwargs)

    def plot_tree_backward(self, V, P, **kwargs):
        self.plot_tree(V, P, **kwargs)

    def plot_path(self, **kwargs):
        path = np.array(self.path)
        plt.plot(path[:,0], path[:,1], **kwargs)

class DubinsRRTConnect(RRTConnect):
    """
    Represents a planning problem for the Dubins car, a model of a simple
    car that moves at a constant speed forward and has a limited turning
    radius. We will use this v0.9.2 of the package at
    https://github.com/AndrewWalker/pydubins/blob/0.9.2/dubins/dubins.pyx
    to compute steering distances and steering trajectories. In particular,
    note the functions dubins.path_length and dubins.path_sample (read
    their documentation at the link above). See
    http://planning.cs.uiuc.edu/node821.html
    for more details on how these steering trajectories are derived.
    """
    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, obstacles, turning_radius):
        self.turning_radius = turning_radius
        super(self.__class__, self).__init__(statespace_lo, statespace_hi, x_init, x_goal, obstacles)

    def reverse_heading(self, x):
        """
        Reverses the heading of a given pose.
        Input: x (np.array [3]): Dubins car pose
        Output: x (np.array [3]): Pose with reversed heading
        """
        theta = x[2]
        if theta < np.pi:
            theta_new = theta + np.pi
        else:
            theta_new = theta - np.pi
        return np.array((x[0], x[1], theta_new))

    def find_nearest_forward(self, V, x):
        ########## Code starts here ##########
        from dubins import path_length
        path_lengths = np.zeros(len(V))
        for i in range(len(V)):
            path_lengths[i] = path_length(V[i,:],x,self.turning_radius)
        return np.argmin(path_lengths)
        ########## Code ends here ##########

    def find_nearest_backward(self, V, x):
        ########## Code starts here ##########
        from dubins import path_length
        path_lengths = np.zeros(len(V))
        for i in range(len(V)):
            #reverse the heading, since backwards
            path_lengths[i] = path_length(self.reverse_heading(V[i,:]),self.reverse_heading(x),self.turning_radius)
        return np.argmin(path_lengths)
        ########## Code ends here ##########

    def steer_towards_forward(self, x1, x2, eps):
        ########## Code starts here ##########
        """
        from dubins import path_sample, path_length
        #div eps by 10 b/c 10 steps, unless premature ending
        configs = path_sample(x1,x2,1.001*self.turning_radius,eps/10)
        if len(configs[0]) < 10:
            x_new = np.array(configs[0][len(configs[0])-1])
        else:
            x_new = np.array(configs[0][9])
        
        #now cover the condition if we are within eps of goal so we can actually reach it
        if path_length(x_new,x1,self.turning_radius) < eps:
            return x2
        else:
            return x_new
        """
        
        from dubins import path_sample, path_length
        configs = path_sample(x1,x2,1.001*self.turning_radius,eps)
        
        if len(configs[0]) == 0:
            return x2
        if len(configs[0]) < 2:
            x_new = np.array(configs[0][0])
        else:
            x_new = np.array(configs[0][1])
        
        #now cover the condition if we are within eps of goal so we can actually reach it
        if path_length(x_new,x1,self.turning_radius) < eps:
            return x2
        else:
            return x_new
        ########## Code ends here ##########

    def steer_towards_backward(self, x1, x2, eps):
        ########## Code starts here ##########
        from dubins import path_sample, path_length
        x1 = self.reverse_heading(x1)
        x2 = self.reverse_heading(x2)
        configs = path_sample(x2,x1,1.001*self.turning_radius,eps)
        
        if len(configs[0]) == 0:
            return self.reverse_heading(x2)
        if len(configs[0]) < 2:
            x_new = np.array(configs[0][0])
        else:
            x_new = np.array(configs[0][1])
                   
        #now cover the condition if we are within eps of goal so we can actually reach it
        if path_length(x_new,x2,self.turning_radius) < eps:
            return self.reverse_heading(x1)
        else:
            return self.reverse_heading(x_new)
        ########## Code ends here ##########

    def is_free_motion(self, obstacles, x1, x2, resolution = np.pi/6):
        pts = path_sample(x1, x2, self.turning_radius, self.turning_radius*resolution)[0]
        pts.append(x2)
        for i in range(len(pts) - 1):
            for line in obstacles:
                if line_line_intersection([pts[i][:2], pts[i+1][:2]], line):
                    return False
        return True

    def plot_tree(self, V, P, resolution = np.pi/24, **kwargs):
        line_segments = []
        for i in range(V.shape[0]):
            if P[i] >= 0:
                pts = path_sample(V[P[i],:], V[i,:], self.turning_radius, self.turning_radius*resolution)[0]
                pts.append(V[i,:])
                for j in range(len(pts) - 1):
                    line_segments.append((pts[j], pts[j+1]))
        plot_line_segments(line_segments, **kwargs)

    def plot_tree_backward(self, V, P, resolution = np.pi/24, **kwargs):
        line_segments = []
        for i in range(V.shape[0]):
            if P[i] >= 0:
                pts = path_sample(V[i,:], V[P[i],:], self.turning_radius, self.turning_radius*resolution)[0]
                pts.append(V[P[i],:])
                for j in range(len(pts) - 1):
                    line_segments.append((pts[j], pts[j+1]))
        plot_line_segments(line_segments, **kwargs)

    def plot_path(self, resolution = np.pi/24, **kwargs):
        pts = []
        path = np.array(self.path)
        for i in range(path.shape[0] - 1):
            pts.extend(path_sample(path[i], path[i+1], self.turning_radius, self.turning_radius*resolution)[0])
        plt.plot([x for x, y, th in pts], [y for x, y, th in pts], **kwargs)

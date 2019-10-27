import numpy as np
import numpy.linalg as la
import numpy.random as npr
from matrixmath import is_pos_def,vec,mdot,solveb
import matplotlib.pyplot as plt
from namedlist import namedlist
from copy import copy

DynamicsDataInput  = namedlist("DynamicsDataInput", "F H Q R")
InitialCondition   = namedlist("InitialCondition", "x_mean0 x_covr0")
UnknownParams      = namedlist("UnknownParams", "Q, R")
LeastSquaresInput  = namedlist("LeastSquaresInput", "kronA kronB L")
LeastSquaresOutput = namedlist("LeastSquaresOutput", "Q_est R_est")

def example_system():
    # LTI system dynamics
    F = np.array([[0.8, 0.2, 0.0],
                  [0.3, 0.5, 0.0],
                  [0.1, 0.9, 0.7]])    
    H = np.array([[1,0,1],
                  [0,1,0]])    
    Q = np.array([[3.0, 0.2, 0.0],
                  [0.2, 2.0, 0.0],
                  [0.0, 0.0, 7.5]]) 
    R = np.array([[5,0],
                  [0,4]]) 
    
    # Initial state mean and covariance estimates
    x_mean0 = np.zeros(3)
    x_covr0 = np.eye(3)
    
    return DynamicsDataInput(F,H,Q,R), InitialCondition(x_mean0,x_covr0)


class AdaptiveKalmanFilter():
    """
    Class representing operations & data needed to perform adaptive KF updates
    """ 
    
    def __init__(self, dynamics_data_input, initial_condition, unknown_params, max_iter):
        """
        Constructor function that initializes simulation data
        Input Parameters:
        max_iter             : Number of maximum filter iterations
        covarianceSelectFlag: Flag to estimate Q or R or both        
        """
        self.F, self.H, self.Q, self.R = dynamics_data_input
        self.n = self.F.shape[0]
        self.m = self.H.shape[0]        
        self.x_mean0, self.x_covr0 = initial_condition
        self.ell = self.check_observability()
        self.p, self.Mopi = self.get_buffer_size()    
        self.max_iter = max_iter 
        self.unknown_params = unknown_params
                    
    
    def obsvp(self, p):
        """
        Returns a p-step observability matrix
        Input Parameters:
        p: Dimension of the required observability matrix
        """
        F = self.F
        H = self.H
        n = self.n
        m = self.m
        
        # Build the observability matrix
        O = np.zeros([m*p,n])
        O[0:m] = np.copy(H)
        for k in range(1,p):
            O[m*k:m*(k+1)] = mdot(O[m*(k-1):m*k],F)
        return O
    
        
    def check_observability(self):
        """
        Checks the observability criterion for system matrices F, H and throws
        an error if system is not observable.
        """        
        
        n = self.n
        O = self.obsvp(n)
        observable = la.matrix_rank(O) == n        
        if observable:
            ell = copy(self.n)         
        else:
            # TODO: generalize to the detectable case
            raise Exception("System is not observable, reformulate")
        return ell

    
    def get_buffer_size(self):
        """
        Returns the size of the measurement history buffer & corresponding 
        observability matrix
        """
        
        n = self.n
        Mo = 0
        p = 0
        while not la.matrix_rank(Mo) == n: # valid only if system is observable
            p += 1
            Mo = self.obsvp(p)
        # Compute the pseudoinverse of Mo
        Mopi = la.pinv(Mo)
        return p, Mopi
        
    
    def lse(self, least_squares_input):
        """
        Performs Least Squares Estimation with input data provided and returns 
        the least squares estimate        
        """
        
        # Unpack the least_squares_input        
        kronA = least_squares_input.kronA
        kronB = least_squares_input.kronB
        L     = least_squares_input.L     
        
        # Unpack the required dynamics_data
        Q   = self.Q
        R   = self.R
        n   = self.n
        m   = self.m
        ell = self.ell
        
        Q_est = np.copy(Q) 
        R_est = np.copy(R)
        
        if self.unknown_params.Q and self.unknown_params.R:         
            S         = np.hstack([kronA,kronB]) 
            vecTheta  = mdot(la.pinv(S),vec(L))
            vecQ_est  = vecTheta[0:ell**2]
            vecR_est  = vecTheta[ell**2:]
            Q_est = np.reshape(vecQ_est,[n,n])
            R_est = np.reshape(vecR_est,[m,m])            
        elif not self.unknown_params.Q and self.unknown_params.R:            
            S         = np.copy(kronB)
            vecCW     = mdot(kronA,vec(Q))
            vecR_est  = mdot(la.pinv(S),vec(L)-vecCW)
            R_est = np.reshape(vecR_est,[m,m])
        elif self.unknown_params.Q and not self.unknown_params.R:             
            S         = np.copy(kronA)
            vecCV     = mdot(kronB,vec(R))
            vecQ_est  = mdot(la.pinv(S),vec(L)-vecCV)
            Q_est = np.reshape(vecQ_est,[n,n])
            
        return LeastSquaresOutput(Q_est, R_est) 
    
    
    def run(self):
        """
        Perform adaptive Kalman filter iterations
        """
        
        T       = self.max_iter
        F       = self.F
        H       = self.H
        Q       = self.Q
        R       = self.R
        n       = self.n
        m       = self.m        
        ell     = self.ell
        p       = self.p       
        Mopi    = self.Mopi
        x_mean0 = self.x_mean0
        x_covr0 = self.x_covr0
        
        # Preallocate history data arrays        
        x_hist      = np.full((T+1,n),np.nan)
        y_hist      = np.full((T+1,m),np.nan)
        x_pre_hist  = np.full((T+1,n),np.nan)
        x_post_hist = np.full((T+1,n),np.nan)
        P_pre_hist  = np.full((T+1,n,n),np.nan)
        P_post_hist = np.full((T+1,n,n),np.nan)
        K_hist      = np.full((T+1,n,m),np.nan)
        Q_est_hist  = np.full((T+1,n,n),np.nan)
        R_est_hist  = np.full((T+1,m,m),np.nan)
        
        # Initialize the iterates
        x_post         = x_mean0
        P_post         = 100*np.eye(n)
        Q_est          = 20*np.eye(n)
        R_est          = 10*np.eye(m)
        L              = np.zeros([n,n]) 
        x              = npr.multivariate_normal(x_mean0,x_covr0)          
        x_hist[0]      = x
        x_post_hist[0] = x_post
        P_post_hist[0] = P_post
        
        # Perform dynamic adaptive Kalman filter updates  
        for k in range(T):              
            # Print the iteration number
            print("k = %9d / %d"%(k+1,T))
            
            # Generate a new multivariate gaussian measurement noise
            v = npr.multivariate_normal(np.zeros(m),R)
            
            # Collect new measurement      
            y = mdot(H,x) + v
            
            # Store the measurement history
            y_hist[k] = y
            
            # Noise covariance estimation
            if k > p-1:                
                # Collect measurement till 'k-1' time steps
                Yold = vec(y_hist[np.arange(k-1,k-p-1,-1)].T)
                
                # Collect measurement till 'k' time steps
                Ynew = vec(y_hist[np.arange(k,k-p,-1)].T)
                
                # Formulating Linear Stationary Time Series
                Z = mdot(Mopi,Ynew) - mdot(F,Mopi,Yold)
                
                # Recursive Covariance Unbiased Estimator
                L = ((k-1)/k)*L + (1/k)*np.outer(Z,Z)  
                
                # Initialize the Stacked H matrix as Mw
                Mw = np.zeros([m*p,ell*(p-1)])
                Mw[0:m,0:ell] = np.copy(H)
                
                # Feed the values into Mw
                for j in range(1,p-1):
                    Mw[0:m,ell*j:ell*(j+1)] = mdot(Mw[0:m,ell*(j-1):ell*j],F)
                for i in range(1,p-1):
                    Mw[m*i:m*(i+1),ell*i:] = Mw[0:m,0:ell*(p-i-1)]            
                
                # Product matrix: Mopi*Mw                
                MopiMw = mdot(Mopi,Mw)
                
                # Form the 'script' A matrix as Anew - Aold
                Anew = np.hstack([MopiMw, np.eye(ell)])
                Aold = np.hstack([np.zeros([ell,ell]),mdot(F,MopiMw)])
                A    = Anew - Aold                 
                Alr  = np.fliplr(A)            
                
                # Form the 'script' B matrix as Bnew - Bold
                Bnew = np.hstack([Mopi,np.zeros([ell,m])])
                Bold = np.hstack([np.zeros([ell,m]),mdot(F,Mopi)])
                B    = Bnew - Bold
                Blr  = np.fliplr(B)
                
                # Get the A_i and B_i sequences
                Ai = np.zeros([p,ell,ell])
                Bi = np.zeros([p+1,ell,m])
                for i in range(p+1):    
                    if i < p:
                        Ai[i] = Alr[:,ell*i:ell*(i+1)]
                    Bi[i] = Blr[:,m*i:m*(i+1)] 
                
                # Form Kronecker producs of the A_i's and B_i's
                kronA = np.zeros([ell**2,ell**2])
                kronB = np.zeros([ell**2,m**2])            
                for i in range(p+1):
                    if i < p:
                        kronA += np.kron(Ai[i],Ai[i])
                    kronB += np.kron(Bi[i],Bi[i])
                    
                # Pack the data required for least squares estimation
                least_squares_input = LeastSquaresInput(kronA, kronB, L)
                
                # Get the Least Squares estimate of selected covariances
                least_squares_output = self.lse(least_squares_input)
                
                # Unpack the least_squares_output
                Q_est_new = least_squares_output.Q_est
                R_est_new = least_squares_output.R_est
                
                # Check positive semidefiniteness of estimated covariances
                if is_pos_def(Q_est_new):
                    Q_est = np.copy(Q_est_new)
                if is_pos_def(R_est_new):
                    R_est = np.copy(R_est_new)
                
            # Update the covariance estimate history
            Q_est_hist[k] = Q_est
            R_est_hist[k] = R_est
                          
            ## Update using standard Kalman filter equations
            # Calculate the a priori state estimate
            x_pre  = mdot(F,x_post)
            # Calculate the a priori error covariance estimate
            P_pre  = mdot(F,P_post,F.T) + Q_est
            # Calculate the Kalman gain
            K      = solveb(mdot(P_pre,H.T),mdot(H,P_pre,H.T)+R_est)
            # Calculate the a posteriori state estimate
            x_post = x_pre + mdot(K,y-mdot(H,x_pre))
            # Calculate the a posteriori error covariance estimate
            IKH    = np.eye(n) - mdot(K,H)            
            P_post = mdot(IKH,P_pre,IKH.T) + mdot(K,R,K.T)
            
            # Store the histories
            x_pre_hist[k+1]  = x_pre
            x_post_hist[k+1] = x_post
            P_pre_hist[k+1]  = P_pre
            P_post_hist[k+1] = P_post
            K_hist[k+1]      = K
                            
            ## True system updates (true state transition and measurement)
            # Generate the process noise
            w = npr.multivariate_normal(np.zeros(n),Q)
            # State update equation
            x = mdot(F,x) + w      
            # Store the true state history
            x_hist[k+1] = x
    
        # Tie up loose ends
        y_hist[-1] = y
        x_pre_hist[0] = x_post_hist[0]
        P_pre_hist[0] = P_post_hist[0]
        K_hist[0] = K_hist[1]
        Q_est_hist[-1] = Q_est
        R_est_hist[-1] = R_est 
        
        # Set the plot parameters
        self.plot_params = [Q, R, Q_est_hist, R_est_hist, x_hist, x_pre_hist, x_post_hist, P_pre_hist, P_post_hist]
        
    
    def plot_data(self):
        """
        Plots the difference of estimated and true covariance matrices
        to visualize the adaptive KF convergence        
        """
        # Get the required plotting data
        T           = self.max_iter
        k_hist      = np.arange(T+1) 
        Q           = self.plot_params[0]
        R           = self.plot_params[1]
        Q_est_hist  = self.plot_params[2]
        R_est_hist  = self.plot_params[3]
        
        
        # Plot convergence of estimated covariances to their true covariances
        fig,ax = plt.subplots(2)
        ax[0].step(k_hist,la.norm(Q_est_hist-Q,ord=2,axis=(1,2)))
        ax[0].set_ylabel("Norm of Q error")
        ax[1].step(k_hist,la.norm(R_est_hist-R,ord=2,axis=(1,2)))
        ax[1].set_ylabel("Norm of R error")
        ax[1].set_xlabel("Time step (k)")
        ax[0].semilogx()
        ax[0].semilogy()
        ax[1].semilogx()
        ax[1].semilogy()
        plt.show()
        
#        x_hist      = self.plot_params[4]
#        x_pre_hist  = self.plot_params[5]
#        x_post_hist = self.plot_params[6]
#        P_pre_hist  = self.plot_params[7]
#        P_post_hist = self.plot_params[8]        
#        # Plot the true, apriori, aposteriori states
#        fig,ax = plt.subplots(n)
#        for i in range(n):
#            ax[i].step(k_hist,x_hist[:,i])
#            ax[i].step(k_hist,x_pre_hist[:,i])
#            ax[i].step(k_hist,x_post_hist[:,i])
#            ax[i].set_ylabel("State %d"%(i+1))
#            ax[i].legend(["True","A priori","A posteriori"])
#        ax[-1].set_xlabel("Time index (k)")
#        
#        fig,ax = plt.subplots()
#        ax.step(k_hist,la.norm(x_hist-x_pre_hist,axis=1))
#        ax.step(k_hist,la.norm(x_hist-x_post_hist,axis=1))
#        ax.legend(["A priori","A posteriori"])
#        ax.set_xlabel("Time index (k)")
#        ax.set_ylabel("Norm of error")
#        
#        fig,ax = plt.subplots()
#        plt.step(k_hist,np.trace(P_pre_hist,axis1=1,axis2=2))
#        plt.step(k_hist,np.trace(P_post_hist,axis1=1,axis2=2))
#        plt.legend(["A priori","A posteriori"])
#        plt.xlabel("Time index (k)")
#        plt.ylabel("Trace of error covariance")
        
###############################################################################

if __name__ == '__main__':    
    plt.close('all')
    npr.seed(2)   
    
    # Define problem parameters
    dynamics_data_input, initial_condition = example_system()
    unknown_params = UnknownParams(Q=False,R=True)
    
    # Initialize the adaptive Kalman filter
    akf = AdaptiveKalmanFilter(dynamics_data_input, initial_condition, unknown_params, max_iter=1000)    
    
    # Perform adaptive Kalman filter iterations
    akf.run()
    
    # Plot covariance estimates
    akf.plot_data()
   

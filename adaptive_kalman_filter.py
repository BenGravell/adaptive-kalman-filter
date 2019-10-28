import numpy as np
import numpy.linalg as la
import numpy.random as npr
from matrixmath import is_pos_def,vec,mdot,solveb
import matplotlib.pyplot as plt
from namedlist import namedlist
from copy import copy


Dynamics      = namedlist("Dynamics", "F H Q R")
InitCond      = namedlist("InitCond", "x_mean x_covr Q_est R_est L")
UnknownParams = namedlist("UnknownParams", "Q, R")
DataHist      = namedlist("DataHist","T Q_est R_est x x_pre x_post P_pre P_post")


class AdaptiveKalmanFilter():
    """
    Class representing operations & data needed to perform adaptive KF updates
    """ 
    def __init__(self, dynamics, init_cond, unknown_params):
        """
        Constructor method for the AdaptiveKalmanFilter class
        Input Parameters:
        dynamics:       Tuple-like with F, H, Q, R matrices
        init_cond:      Tuple-like with x_mean0 vector and x_covr0 matrix
        unknown_params: Tuple with boolean for Q, R estimation
        """
        self.F, self.H, self.Q, self.R = dynamics
        self.n = self.F.shape[0]
        self.m = self.H.shape[0]        
        self.x_mean0,self.x_covr0,self.Q_est0,self.R_est0,self.L0 = init_cond
        self.ell = self.check_observability()
        self.p, self.Mopi, self.kronA, self.kronB = self.get_buffer_size()
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
            # TODO: generalize to the detectable case i.e. 
            #       by generalizing this method to return the matrices which
            #       split the system into observable and unobservable spaces
            #       and the corresponding dimension ell
            raise Exception("System is not observable, reformulate")
        return ell
    
    def get_buffer_size(self):
        """
        Returns the size of the measurement history buffer & corresponding 
        observability matrix
        """        
        n = self.n
        m = self.m
        ell = self.ell
        F = self.F
        H = self.H
        Mo = 0
        p = 0
        while not la.matrix_rank(Mo) == n: # valid only if system is observable
            p += 1
            Mo = self.obsvp(p)
        Mopi = la.pinv(Mo)

        # Build the stacked H matrix as Mw
        if p > 1:
            Mw = np.zeros([m*p,ell*(p-1)])
            Mw[0:m,0:ell] = np.copy(H)
            for j in range(1,p-1):
                Mw[0:m,ell*j:ell*(j+1)] = mdot(Mw[0:m,ell*(j-1):ell*j],F)
            for i in range(1,p-1):
                Mw[m*i:m*(i+1),ell*i:] = Mw[0:m,0:ell*(p-i-1)]

            # Product matrix: Mopi*Mw
            MopiMw = mdot(Mopi,Mw)

            # Form the 'script' A matrix as Anew - Aold
            Anew = np.hstack([MopiMw, np.eye(ell)])
            Aold = np.hstack([np.zeros([ell,ell]),mdot(F,MopiMw)])
            A = Anew - Aold

            # Form the 'script' B matrix as Bnew - Bold
            Bnew = np.hstack([Mopi,np.zeros([ell,m])])
            Bold = np.hstack([np.zeros([ell,m]),mdot(F,Mopi)])
            B = Bnew - Bold
        else:
            A = np.eye(ell)
            B = np.hstack([Mopi, -mdot(F,Mopi)])

        # Get the A_i and B_i sequences
        Ai = np.zeros([p,ell,ell])
        Bi = np.zeros([p+1,ell,m])
        for i in range(p+1):
            if i > 0:
                Ai[p-i] = A[:,ell*(i-1):ell*(i)]
            Bi[p-i] = B[:,m*(i):m*(i+1)]

        # Form Kronecker products of the A_i's and B_i's
        kronA = np.zeros([ell**2,ell**2])
        kronB = np.zeros([ell**2,m**2])
        for i in range(p+1):
            if i < p:
                kronA += np.kron(Ai[i],Ai[i])
            kronB += np.kron(Bi[i],Bi[i])

        return p, Mopi, kronA, kronB
        
    def lse(self, L):
        """
        Performs Least Squares Estimation with input data provided and returns 
        the least squares estimate        
        """                
        # Unpack the required dynamics data
        Q = self.Q
        R = self.R
        n = self.n
        m = self.m
        ell = self.ell
        kronA = self.kronA
        kronB = self.kronB
        
        # Perform least-squares estimation
        Q_est = np.copy(Q) 
        R_est = np.copy(R)        
        if self.unknown_params.Q and self.unknown_params.R:         
            S = np.hstack([kronA,kronB]) 
            vecTheta = mdot(la.pinv(S),vec(L))
            vecQ_est = vecTheta[0:ell**2]
            vecR_est = vecTheta[ell**2:]
            Q_est = np.reshape(vecQ_est,[n,n])
            R_est = np.reshape(vecR_est,[m,m])            
        elif not self.unknown_params.Q and self.unknown_params.R:            
            S = np.copy(kronB)
            vecCW = mdot(kronA,vec(Q))
            vecR_est = mdot(la.pinv(S),vec(L)-vecCW)
            R_est = np.reshape(vecR_est,[m,m])
        elif self.unknown_params.Q and not self.unknown_params.R:             
            S = np.copy(kronA)
            vecCV = mdot(kronB,vec(R))
            vecQ_est = mdot(la.pinv(S),vec(L)-vecCV)
            Q_est = np.reshape(vecQ_est,[n,n])              
        return Q_est, R_est 
    
    def run(self, T):
        """
        Perform adaptive Kalman filter iterations
        Input Parameters:
        T: Integer of maximum filter iterations
        """        
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
        Q_est0  = self.Q_est0
        R_est0  = self.R_est0
        L0      = self.L0
        
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
        P_post         = x_covr0
        Q_est          = Q_est0
        R_est          = R_est0
        L              = L0
        x              = npr.multivariate_normal(x_mean0,x_covr0)          
        x_hist[0]      = x
        x_post_hist[0] = x_post
        P_post_hist[0] = P_post        
        
        # Perform dynamic adaptive Kalman filter updates  
        for k in range(T):              
            # Print the iteration number
            print("k = %9d / %d"%(k+1,T))            
            # Generate a new multivariate Gaussian measurement noise
            v = npr.multivariate_normal(np.zeros(m),R)            
            # Collect and store a new measurement      
            y = mdot(H,x) + v
            y_hist[k] = y            
            # Noise covariance estimation
            if k > p-1:                
                # Collect measurement till 'k-1' time steps
                Yold = vec(y_hist[np.arange(k-1,k-p-1,-1)].T)
                
                # Collect measurement till 'k' time steps
                Ynew = vec(y_hist[np.arange(k,k-p,-1)].T)
                
                # Formulate a linear stationary time series
                Z = mdot(Mopi,Ynew) - mdot(F,Mopi,Yold)
                
                # Recursive covariance unbiased estimator
                L = ((k-1)/k)*L + (1/k)*np.outer(Z,Z)  

                # Get the least squares estimate of selected covariances
                Q_est_new, R_est_new = self.lse(L)
                
                # Check positive semidefiniteness of estimated covariances
                if is_pos_def(Q_est_new):
                    Q_est = np.copy(Q_est_new)
                if is_pos_def(R_est_new):
                    R_est = np.copy(R_est_new)
                
            # Update the covariance estimate history
            Q_est_hist[k] = Q_est
            R_est_hist[k] = R_est
                          
            ## Update state estimates using standard Kalman filter equations
            # Calculate the a priori state estimate
            x_pre = mdot(F,x_post)
            
            # Calculate the a priori error covariance estimate
            P_pre = mdot(F,P_post,F.T) + Q_est
            
            # Calculate the Kalman gain
            K = solveb(mdot(P_pre,H.T),mdot(H,P_pre,H.T)+R_est)
            
            # Calculate the a posteriori state estimate
            x_post = x_pre + mdot(K,y-mdot(H,x_pre))
            
            # Calculate the a posteriori error covariance estimate
            IKH = np.eye(n) - mdot(K,H)            
            P_post = mdot(IKH,P_pre,IKH.T) + mdot(K,R,K.T)
            
            # Store the histories
            x_pre_hist[k+1]  = x_pre
            x_post_hist[k+1] = x_post
            P_pre_hist[k+1]  = P_pre
            P_post_hist[k+1] = P_post
            K_hist[k+1]      = K
                            
            ## True system updates (true state transition and measurement)
            # Generate process noise
            w = npr.multivariate_normal(np.zeros(n),Q)

            # Update and store the state
            x = mdot(F,x) + w      
            x_hist[k+1] = x
    
        # Tie up loose ends
        y_hist[-1] = y
        x_pre_hist[0] = x_post_hist[0]
        P_pre_hist[0] = P_post_hist[0]
        K_hist[0] = K_hist[1]
        Q_est_hist[-1] = Q_est
        R_est_hist[-1] = R_est 
        
        # Return data history
        data_hist = DataHist(T, Q_est_hist, R_est_hist, x_hist, x_pre_hist,
                                  x_post_hist, P_pre_hist, P_post_hist)
        return data_hist

    
def plot_data(akf,data_hist):
    """
    Plots the difference of estimated and true covariance matrices
    to visualize the adaptive KF convergence        
    """
    # Get the required plotting data
    n           = akf.n
    Q           = akf.Q
    R           = akf.R
    T           = data_hist.T
    k_hist      = np.arange(T+1) 
    Q_est_hist  = data_hist.Q_est
    R_est_hist  = data_hist.R_est
    x_hist      = data_hist.x
    x_pre_hist  = data_hist.x_pre
    x_post_hist = data_hist.x_post
    P_pre_hist  = data_hist.P_pre
    P_post_hist = data_hist.P_post
    
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
    ax[0].set_title("Noise covariance error vs time")

    if T <= 1000:
        # Plot the true, apriori, aposteriori states
        fig,ax = plt.subplots(n)
        for i in range(n):
            ax[i].step(k_hist,x_hist[:,i])
            ax[i].step(k_hist,x_pre_hist[:,i])
            ax[i].step(k_hist,x_post_hist[:,i])
            ax[i].set_ylabel("State %d"%(i+1))
            ax[i].legend(["True","A priori","A posteriori"])
        ax[-1].set_xlabel("Time index (k)")
        ax[0].set_title("State vs time")

        # Plot the state estimation error
        fig,ax = plt.subplots()
        ax.step(k_hist,la.norm(x_hist-x_pre_hist,axis=1))
        ax.step(k_hist,la.norm(x_hist-x_post_hist,axis=1))
        ax.legend(["A priori","A posteriori"])
        ax.set_xlabel("Time index (k)")
        ax.set_ylabel("Norm of error")
        plt.title("Actual error vs time")

        # Plot the assumed error covariance trace
        fig,ax = plt.subplots()
        plt.step(k_hist,np.trace(P_pre_hist,axis1=1,axis2=2))
        plt.step(k_hist,np.trace(P_post_hist,axis1=1,axis2=2))
        plt.legend(["A priori","A posteriori"])
        plt.xlabel("Time index (k)")
        plt.ylabel("Trace of error covariance")
        plt.title("Assumed error vs time")
    
    plt.show()


def example_system():
    # Linear time-invariant system dynamics
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
    
    # Initial estimates
    x_mean = np.zeros(3)
    x_covr = np.eye(3)    
    Q_est  = 20*np.eye(3)
    R_est  = 10*np.eye(2)
    L      = np.zeros([3,3])
    
    return Dynamics(F,H,Q,R), InitCond(x_mean,x_covr,Q_est,R_est,L)


def example_system2():
    # Linear time-invariant system dynamics
    F = np.array([[0.8, 0.2],
                  [0.3, 0.5]])
    H = np.array([[1,0],
                  [0,0.2]])
    Q = np.array([[3.0, 0.2],
                  [0.2, 2.0]])
    R = np.array([[5,0],
                  [0,4]])

    # Initial estimates
    x_mean = np.zeros(2)
    x_covr = np.eye(2)
    Q_est  = 20*np.eye(2)
    R_est  = 10*np.eye(2)
    L      = np.zeros([2,2])

    return Dynamics(F,H,Q,R), InitCond(x_mean,x_covr,Q_est,R_est,L)


if __name__ == '__main__':    
    # Initialize the session
    plt.close('all')
    npr.seed(1)   
    
    # Define problem parameters
    # dynamics, init_cond = example_system()
    dynamics, init_cond = example_system2()
    unknown_params = UnknownParams(Q=False, R=True)
    # unknown_params = UnknownParams(Q=True, R=False)
    
    # Initialize the adaptive Kalman filter
    akf = AdaptiveKalmanFilter(dynamics, init_cond, unknown_params)    
    
    # Perform adaptive Kalman filter iterations
    data_hist = akf.run(T=10000)
    
    # Plot covariance estimates
    plot_data(akf,data_hist)
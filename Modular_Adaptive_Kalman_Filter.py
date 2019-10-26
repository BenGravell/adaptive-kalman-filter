import numpy as np
import numpy.linalg as la
import numpy.random as npr
from matrixmath import is_pos_def,vec,mdot,specrad,sympart,minsv,solveb
import matplotlib.pyplot as plt

###############################################################################
###############################################################################

class Adaptive_Kalman_Filter():
    """
    Class representing operations & data needed to perform adaptive KF updates
    """ 
    
    def __init__(self, maxIter, covarianceSelectFlag):
        """
        Constructor function that initializes simulation data
        Input Parameters:
        maxIter             : Number of maximum filter iterations
        covarianceSelectFlag: Flag to estimate Q or R or both        
        """
        self.T             = maxIter 
        self.covSelectFlag = covarianceSelectFlag
        
        
    ###########################################################################
        
    def GetDynamicsData(self):
        """
        Loads the System Dynamics data into the class object
        """
        # System model parameters    
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
        
        n = F.shape[0]
        m = H.shape[0]   
        
        # Initial state mean and covariance estimates
        x_mean0 = np.zeros(n)
        x_covr0 = np.eye(n)
        
        # Form the dynamicsData list 
        self.dynamicsData = [F,H,Q,R,n,m,x_mean0,x_covr0]   
        
        # Check Observability and update the dynamicsData with ell
        self.CheckObservability()
        
        # Get the buffer size
        self.GetBufferSize()
        
        # Updated dynamicsData is [F,H,Q,R,n,m,x_mean0,x_covr0,ell,p,Mopi]   
            
    ###########################################################################
    
    def GetObservabilityMatrix(self, p):
        """
        Returns a p-step observability matrix
        Input Parameters:
        p: Dimension of the required observability matrix
        """
        F = self.dynamicsData[0]
        H = self.dynamicsData[1]
        n = self.dynamicsData[4]
        m = self.dynamicsData[5] 
        # Initialize the observability matrix
        observabilityMatrix = np.zeros([m*p,n])
        observabilityMatrix[0:m] = np.copy(H)
        for k in range(1,p):
            observabilityMatrix[m*k:m*(k+1)] = mdot(observabilityMatrix[m*(k-1):m*k],F)
        return observabilityMatrix
    
    ###########################################################################
        
    def CheckObservability(self):
        """
        Checks the observability criterion for system matrices F, H and throws
        an error if system is not observable.
        """        
        n = self.dynamicsData[4]
        # Get the Observability Matrix
        observabilityMatrix = self.GetObservabilityMatrix(n)
        # Check its rank
        observable = la.matrix_rank(observabilityMatrix) == n
        # if not observable - throw an error
        if observable:
            # Set ell = n
            self.dynamicsData.append(n)            
        else:
            raise Exception("System is not observable, reformulate")
            
    ###########################################################################
    
    def GetBufferSize(self):
        """
        Returns the size of the measurement history buffer & corresponding 
        observability matrix
        """
        n = self.dynamicsData[4]
        Mo = 0
        p = 0
        while not la.matrix_rank(Mo) == n: # valid only if system is observable
            p += 1
            Mo = self.GetObservabilityMatrix(p)
        # Compute the inverse of Mo
        Mopi = la.pinv(Mo)
        
        # Update the dynamicsData with p, Mopi
        self.dynamicsData.append(p)        
        self.dynamicsData.append(Mopi)
        
    ###########################################################################
    
    def PerformLeastSquaresEstimation(self, leastSquaresInput):
        """
        Performs Least Squares Estimation with input data provided and returns 
        the least squares estimate
        Input Parameters:
        leastSquaresInput: measurements, sensor matrix, covarianceSelect flag
        """
        
        # Unpack the leastSquaresInput        
        kronA                = leastSquaresInput[0]
        kronB                = leastSquaresInput[1]
        L                    = leastSquaresInput[2]        
        
        # Unpack the required dynamicsData
        Q   = self.dynamicsData[2]
        R   = self.dynamicsData[3]
        n   = self.dynamicsData[4]
        m   = self.dynamicsData[5]
        ell = self.dynamicsData[8]        
        
        if self.covSelectFlag == 1: # Estimate both Unknown Q and R            
            S = np.hstack([kronA,kronB]) 
            vecTheta = mdot(la.pinv(S),vec(L))
            vecQ_est = vecTheta[0:ell**2]
            vecR_est = vecTheta[ell**2:]
            Q_est_new = np.reshape(vecQ_est,[n,n])
            R_est_new = np.reshape(vecR_est,[m,m])            
        elif self.covSelectFlag == 2: # Estimate Unknown R alone - Q is already known            
            S = np.copy(kronB)
            vecCW = mdot(kronA,vec(Q))
            vecR_est = mdot(la.pinv(S),vec(L)-vecCW)
            R_est_new = np.reshape(vecR_est,[m,m])
            Q_est_new = np.copy(Q)            
        elif self.covSelectFlag == 3: # Estimate  Unknown Q alone - R is already known                
            S = np.copy(kronA)
            vecCV = mdot(kronB,vec(R))
            vecQ_est = mdot(la.pinv(S),vec(L)-vecCV)
            Q_est_new = np.reshape(vecQ_est,[n,n])
            R_est_new = np.copy(R)
            
        # Pack the output data and return it
        leastSquaresOutput = [Q_est_new, R_est_new]            
        return leastSquaresOutput
    
    ###########################################################################
    
    def AdaptiveKalmanFilterIteration(self):
        """
        Initializes data structures for Adaptive Kalman Filter iterations
        """
        T    = self.T
        F    = self.dynamicsData[0]
        H    = self.dynamicsData[1]
        Q    = self.dynamicsData[2]
        R    = self.dynamicsData[3]
        n    = self.dynamicsData[4]
        m    = self.dynamicsData[5]        
        ell  = self.dynamicsData[8]
        p    = self.dynamicsData[9]        
        Mopi = self.dynamicsData[10]
        
        # Define the data structures
        
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
        x_post         = self.dynamicsData[6] # x_mean0
        P_post         = 100*np.eye(n)
        Q_est          = 20*np.eye(n)
        R_est          = 10*np.eye(m)
        L              = np.zeros([n,n]) 
        x              = npr.multivariate_normal(self.dynamicsData[6],self.dynamicsData[7]) # (x_mean0, x_covr0)           
        x_hist[0]      = x
        x_post_hist[0] = x_post
        P_post_hist[0] = P_post
        
        # Dynamic adaptive Kalman filter updates  
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
                
                # Kron the A_i's and B_i's
                kronA = np.zeros([ell**2,ell**2])
                kronB = np.zeros([ell**2,m**2])            
                for i in range(p+1):
                    if i < p:
                        kronA += np.kron(Ai[i],Ai[i])
                    kronB += np.kron(Bi[i],Bi[i])
                    
                # Pack the data required for least squares estimation
                leastSquaresInput = [kronA, kronB, L]
                
                # Get the Least Squares estimate of selected covariances
                leastSquaresOutput = self.PerformLeastSquaresEstimation(leastSquaresInput)
                
                # Unpack the leastSquaresOutput
                Q_est_new = leastSquaresOutput[0]
                R_est_new = leastSquaresOutput[1]
                
                # Check positive semidefiniteness of estimated covariances
                if is_pos_def(Q_est_new):
                    Q_est = np.copy(Q_est_new)
                if is_pos_def(R_est_new):
                    R_est = np.copy(R_est_new)
                
                # Update the covariance estimate history
                Q_est_hist[k] = Q_est
                R_est_hist[k] = R_est
                
            else:                
                # If k <= p - 1
                Q_est_hist[k] = Q_est
                R_est_hist[k] = R_est
                          
            ## Update using Standard Kalman filter equations
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
                            
            ## True System updates (true state transition and measurement)
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
        self.plotParams = [Q, R, Q_est_hist, R_est_hist, x_hist, x_pre_hist, x_post_hist, P_pre_hist, P_post_hist]
        
    ###########################################################################
    
    def PlotData(self):
        """
        Plots the Difference of estimated and true covariance matrices
        to visualize the adaptive KF convergence        
        """
        # Get the required plotting data
        T           = self.T
        k_hist      = np.arange(T+1) 
        Q           = self.plotParams[0]
        R           = self.plotParams[1]
        Q_est_hist  = self.plotParams[2]
        R_est_hist  = self.plotParams[3]
        
        
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
        
#        x_hist      = self.plotParams[4]
#        x_pre_hist  = self.plotParams[5]
#        x_post_hist = self.plotParams[6]
#        P_pre_hist  = self.plotParams[7]
#        P_post_hist = self.plotParams[8]        
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
###############################################################################
###############################################################################

def main():    
    
    # Close any existing figure
    plt.close('all')
    npr.seed(2)
    
    # Create the Adaptive_Kalman_Filter Class Object by initizalizng the required data
    # covarianceSelectFlag = 1: Estimate both Q and R
    # covarianceSelectFlag = 2: Estimate R alone, Q is already known
    # covarianceSelectFlag = 3: Estimate Q alone, R is already known    
    adaptiveKF = Adaptive_Kalman_Filter(maxIter=1000, covarianceSelectFlag = 1)
    
    # Perform Adaptive_Kalman_Filter iteration and see convergence
    adaptiveKF.GetDynamicsData()
    adaptiveKF.AdaptiveKalmanFilterIteration()
    
    # Plot the Convergence of Covariance Estimates
    adaptiveKF.PlotData()

###############################################################################

if __name__ == '__main__':
    main()
    
###############################################################################
###############################################################################
###################### END OF THE FILE ########################################
###############################################################################
###############################################################################
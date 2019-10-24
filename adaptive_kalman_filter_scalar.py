import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import numpy.random as npr
from matrixmath import is_pos_def,vec,mdot,specrad,sympart,minsv,solveb
import matplotlib.pyplot as plt


# Generate a p-step observability matrix
def obsvp(F,H,p):
    n = H.shape[1]
    m = H.shape[0]
    M = np.zeros([m*p,n])
    M[0:m] = np.copy(H)
    for k in range(1,p):
        M[m*k:m*(k+1)] = mdot(M[m*(k-1):m*k],F)
    return M


if __name__ == "__main__":
    plt.close('all')
    npr.seed(1)
    
    # System model parameters    
    F = np.array(0.5)    
    H = np.array(1)    
    Q = np.array(3) 
    R = np.array(5) 
    n = 1
    m = 1
    x_mean0 = 0
    x_covr0 = 1

    # System is observable for H != 0
    #obsvp(F,H,p) = H
    M = np.copy(H)
    ell = n
    
    # System only requires a single measurement history
    Mo = np.copy(H)
    p = 1
    
    # Initialization
    x_post = x_mean0
    P_post = 10
    x = x_mean0+x_covr0**0.5*npr.randn()
    Q_est = 100
    R_est = 10
    L = 0
    
    T = 10000
    x_hist      = np.full(T+1,np.nan)
    y_hist      = np.full(T+1,np.nan)
    x_pre_hist  = np.full(T+1,np.nan)
    x_post_hist = np.full(T+1,np.nan)
    P_pre_hist  = np.full(T+1,np.nan)
    P_post_hist = np.full(T+1,np.nan)
    K_hist      = np.full(T+1,np.nan)
    Q_est_hist  = np.full(T+1,np.nan)
    R_est_hist  = np.full(T+1,np.nan)
    
    x_hist[0] = x
    x_post_hist[0] = x_post
    P_post_hist[0] = P_post
    estimateQ = False
    estimateR = True
    
    # Dynamic adaptive Kalman filter updates    
    for k in range(T):       
        print("k = %9d / %d"%(k+1,T))
        # Collect new measurement      
        v = R**0.5*npr.randn()
        y = H*x + v
        y_hist[k] = y
        
        # Noise covariance estimation
        if k > 0: 
            Yold = y_hist[k-1]
            Ynew = y_hist[k]
            Mopi = 1/Mo
            Z = Mopi*Ynew-F*Mopi*Yold
            L = ((k-1)/k)*L + (1/k)*Z*Z  
            Ai = 1
            Bi = np.array([-F*Mopi,Mopi])     
            kronA = Ai**2
            kronB = Bi[0]**2+Bi[1]**2            
            if estimateQ and estimateR:            
                # Unknown Q and R                      
                S = np.hstack([kronA,kronB])                
                Theta = la.pinv(S)*vec(L)
                Q_est_new = Theta[0]
                R_est_new = Theta[1]       
            elif not estimateQ and estimateR:
                # Unknown R                               
                S = np.copy(kronB)
                CW = kronA*Q
                R_est_new = (L-CW)/S
            elif estimateQ and not estimateR:     
                # Unknown Q                          
                S = np.copy(kronA)        
                CV = kronB*R        
                Q_est_new = (L-CV)/S
            if not estimateQ:
                Q_est_new = np.copy(Q)    
            if not estimateR:
                R_est_new = np.copy(R)                       

            # Check positive semidefiniteness of estimated covariances
            if Q_est_new > 0:
                Q_est = np.copy(Q_est_new)
            if R_est_new > 0:
                R_est = np.copy(R_est_new)            
            Q_est_hist[k]  = Q_est
            R_est_hist[k]  = R_est
        
        # Kalman filter updates (state estimation)
        x_pre  = F*x_post
        P_pre  = F*P_post*F + Q_est
        K      = P_pre*H/(H*P_pre*H+R_est)
        x_post = x_pre + K*(y-H*x_pre)
        IKH    = 1 - K*H
        P_post = IKH*P_pre*IKH + K*R*K
        
        x_pre_hist[k+1]  = x_pre
        x_post_hist[k+1] = x_post
        P_pre_hist[k+1]  = P_pre
        P_post_hist[k+1] = P_post
        K_hist[k+1]      = K        
                
        # System updates (true state transition and measurement)
        w = Q**0.5*npr.randn()
        x = F*x + w             
        x_hist[k+1] = x
            
    plot = True
    if plot:   
        k_hist = np.arange(T+1)    
        
        # fig,ax = plt.subplots()
        # ax.step(k_hist,x_hist)
        # ax.step(k_hist,x_pre_hist)
        # ax.step(k_hist,x_post_hist)
        # ax.set_ylabel("State")
        # ax.legend(["True","A priori","A posteriori"])
        # ax.set_xlabel("Time index (k)")
        #
        # fig,ax = plt.subplots()
        # ax.step(k_hist,np.abs(x_hist-x_pre_hist))
        # ax.step(k_hist,np.abs(x_hist-x_post_hist))
        # ax.legend(["A priori","A posteriori"])
        # ax.set_xlabel("Time index (k)")
        # ax.set_ylabel("Norm of error")
        #
        # fig,ax = plt.subplots()
        # plt.step(k_hist,np.abs(P_pre_hist))
        # plt.step(k_hist,np.abs(P_post_hist))
        # plt.legend(["A priori","A posteriori"])
        # plt.xlabel("Time index (k)")
        # plt.ylabel("Trace of error covariance")
        
        fig,ax = plt.subplots(2)
        ax[0].step(k_hist,np.abs(Q_est_hist-Q))
        ax[0].set_ylabel("Norm of Q error")
        ax[1].step(k_hist,np.abs(R_est_hist-R))
        ax[1].set_ylabel("Norm of R error")
        ax[1].set_xlabel("Time step (k)")
        ax[0].semilogx()
        ax[0].semilogy()
        ax[1].semilogx()
        ax[1].semilogy()
        
        
#        fig,ax = plt.subplots(2)
#        ax[0].step(k_hist,Q*np.ones_like(k_hist))
#        ax[0].step(k_hist,Q_est_hist)        
#        ax[0].set_ylabel("Q")
#        ax[0].legend(["True","Estimate"])
#        ax[1].step(k_hist,R*np.ones_like(k_hist))
#        ax[1].step(k_hist,R_est_hist)
#        ax[1].set_ylabel("R")
#        ax[1].legend(["True","Estimate"])
#        ax[1].set_xlabel("Time step (k)")    
        
        plt.show()
        
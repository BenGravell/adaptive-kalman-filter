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
    npr.seed(2)
    
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
    x_mean0 = np.zeros(n)
    x_covr0 = np.eye(n)

    # Check observability
    M = obsvp(F,H,n)
    observable = la.matrix_rank(M) == n
    if observable:
        ell = n
    else:
        raise Exception("System is not observable, reformulate")
    
    # Find the size of the measurement history buffer
    Mo = 0
    p = 0
    while not la.matrix_rank(Mo)==n: # valid only if system is observable
        p += 1
        Mo = obsvp(F,H,p)
    
    # Initialization
    x_post = x_mean0
    P_post = 100*np.eye(n)
    x = npr.multivariate_normal(x_mean0,x_covr0)
    Q_est = 20*np.eye(n)
    R_est = 10*np.eye(m)
    L = np.zeros([n,n])    
    T = 1000
    
    x_hist      = np.full((T+1,n),np.nan)
    y_hist      = np.full((T+1,m),np.nan)
    x_pre_hist  = np.full((T+1,n),np.nan)
    x_post_hist = np.full((T+1,n),np.nan)
    P_pre_hist  = np.full((T+1,n,n),np.nan)
    P_post_hist = np.full((T+1,n,n),np.nan)
    K_hist      = np.full((T+1,n,m),np.nan)
    Q_est_hist  = np.full((T+1,n,n),np.nan)
    R_est_hist  = np.full((T+1,m,m),np.nan)
    
    x_hist[0] = x
    x_post_hist[0] = x_post
    P_post_hist[0] = P_post
    estimateQ = False
    estimateR = True

    # Dynamic adaptive Kalman filter updates  
    for k in range(T):      
        print("k = %9d / %d"%(k+1,T))
        # Collect new measurement      
        v = npr.multivariate_normal(np.zeros(m),R)
        y = mdot(H,x) + v
        y_hist[k] = y
        
        # Noise covariance estimation
        if k > p-1:
            Yold = vec(y_hist[np.arange(k-1,k-p-1,-1)].T)
            Ynew = vec(y_hist[np.arange(k,k-p,-1)].T)
            Mopi = la.pinv(Mo)
            Z = mdot(Mopi,Ynew)-mdot(F,Mopi,Yold)
            L = ((k-1)/k)*L + (1/k)*np.outer(Z,Z)            
            Mw = np.zeros([m*p,ell*(p-1)])
            Mw[0:m,0:ell] = np.copy(H)
            for j in range(1,p-1):
                Mw[0:m,ell*j:ell*(j+1)] = mdot(Mw[0:m,ell*(j-1):ell*j],F)
            for i in range(1,p-1):
                Mw[m*i:m*(i+1),ell*i:] = Mw[0:m,0:ell*(p-i-1)]            
            MopiMw = mdot(Mopi,Mw)
            Anew = np.hstack([MopiMw, np.eye(ell)])
            Aold = np.hstack([np.zeros([ell,ell]),mdot(F,MopiMw)])
            A = Anew-Aold 
            Alr = np.fliplr(A)            
            Bnew = np.hstack([Mopi,np.zeros([ell,m])])
            Bold = np.hstack([np.zeros([ell,m]),mdot(F,Mopi)])
            B = Bnew-Bold
            Blr = np.fliplr(B)            
            Ai = np.zeros([p,ell,ell])
            Bi = np.zeros([p+1,ell,m])
            for i in range(p+1):    
                if i < p:
                    Ai[i] = Alr[:,ell*i:ell*(i+1)]
                Bi[i] = Blr[:,m*i:m*(i+1)] 
            kronA = np.zeros([ell**2,ell**2])
            kronB = np.zeros([ell**2,m**2])            
            for i in range(p+1):
                if i < p:
                    kronA += np.kron(Ai[i],Ai[i])
                kronB += np.kron(Bi[i],Bi[i])
            
            if estimateQ and estimateR:
                # Unknown Q and R
                S = np.hstack([kronA,kronB]) 
                vecTheta = mdot(la.pinv(S),vec(L))
                vecQ_est = vecTheta[0:ell**2]
                vecR_est = vecTheta[ell**2:]
                Q_est_new = np.reshape(vecQ_est,[n,n])
                R_est_new = np.reshape(vecR_est,[m,m])
            elif not estimateQ and estimateR:
                # Unknown R
                S = np.copy(kronB)
                vecCW = mdot(kronA,vec(Q))
                vecR_est = mdot(la.pinv(S),vec(L)-vecCW)
                R_est_new = np.reshape(vecR_est,[m,m])
            elif estimateQ and not estimateR:     
                # Unknown Q
                S = np.copy(kronA)
                vecCV = mdot(kronB,vec(R))
                vecQ_est = mdot(la.pinv(S),vec(L)-vecCV)
                Q_est_new = np.reshape(vecQ_est,[n,n])
            if not estimateQ:
                Q_est_new = np.copy(Q)    
            if not estimateR:
                R_est_new = np.copy(R)            
            
            # Check positive semidefiniteness of estimated covariances
            if is_pos_def(Q_est_new):
                Q_est = np.copy(Q_est_new)
            if is_pos_def(R_est_new):
                R_est = np.copy(R_est_new)
            
    #        Q_est = psdpart(Q_est_new)
    #        R_est = psdpart(R_est_new)
            Q_est_hist[k] = Q_est
            R_est_hist[k] = R_est
        else:
            Q_est_hist[k] = Q_est
            R_est_hist[k] = R_est

        # Kalman filter updates (state estimation)
        x_pre  = mdot(F,x_post)
        P_pre  = mdot(F,P_post,F.T) + Q_est
        K      = solveb(mdot(P_pre,H.T),mdot(H,P_pre,H.T)+R_est)
        x_post = x_pre + mdot(K,y-mdot(H,x_pre))
        IKH    = np.eye(n)-mdot(K,H)
        P_post = mdot(IKH,P_pre,IKH.T)+mdot(K,R,K.T)
        
        x_pre_hist[k+1]  = x_pre
        x_post_hist[k+1] = x_post
        P_pre_hist[k+1]  = P_pre
        P_post_hist[k+1] = P_post
        K_hist[k+1]      = K
                        
        # System updates (true state transition and measurement)
        w = npr.multivariate_normal(np.zeros(n),Q)
        x = mdot(F,x) + w             
        x_hist[k+1] = x

    # Tie up loose ends
    y_hist[-1] = y
    x_pre_hist[0] = x_post_hist[0]
    P_pre_hist[0] = P_post_hist[0]
    K_hist[0] = K_hist[1]
    Q_est_hist[-1] = Q_est
    R_est_hist[-1] = R_est

    
    plot = True
    if plot:    
        k_hist = np.arange(T+1)    
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
        
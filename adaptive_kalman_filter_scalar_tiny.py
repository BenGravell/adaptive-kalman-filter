import numpy as np, numpy.random as npr
F, H, Q, R = 0.5, 1, 3, 5
x_post, P_post, R_est, L, T = 0, 10, 10, 0, 100000
x = npr.randn()
for k in range(T):
    w,v = Q**0.5*npr.randn(), R**0.5*npr.randn()
    ynew = H*x + v
    if k > 0:
        Mopi = 1/H
        Z = Mopi*ynew-F*Mopi*yold
        L = L*(k-1)/k + Z*Z/k
        Ai, Bi = 1, np.array([-F*Mopi,Mopi])
        kronA, kronB = Ai**2, Bi[0]**2+Bi[1]**2
        S = np.copy(kronB)
        CW = kronA*Q
        R_est_new = (L-CW)/S
        if R_est_new > 0: R_est = R_est_new
    x_pre  = F*x_post
    P_pre  = F*P_post*F + Q
    K      = P_pre*H/(H*P_pre*H+R_est)
    x_post = x_pre + K*(ynew-H*x_pre)
    IKH    = 1 - K*H
    P_post = IKH*P_pre*IKH + K*R*K
    x = F*x + w
    yold = np.copy(ynew)
    print('k = %8d: R_true = %8.6f, R_est = %8.6f' % (k+1,R,R_est))
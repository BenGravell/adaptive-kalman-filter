import numpy as np, numpy.random as npr, matplotlib.pyplot as plt
F, H, Q, R = 0.5, 2, 4, 10
x_post, P_post, R_est, L = 0, 100, 100, 0
x = npr.randn()
N = 10
T = 10000
x_true_history_true     = np.zeros([N,T])
x_true_history_static   = np.zeros([N,T])
x_true_history_adaptive = np.zeros([N,T])
x_post_history_true     = np.zeros([N,T])
x_post_history_static   = np.zeros([N,T])
x_post_history_adaptive = np.zeros([N,T])
R_est_history           = np.zeros([N,T])
k_hist                  = np.arange(T)+1

Mopi    = 1/H
A1      = 1
B1      = -F*Mopi
B2      = Mopi
kronA   = A1**2
kronB   = B1**2+B2**2
S       = np.copy(kronB)
CW      = kronA*Q
print_k = np.logspace(0,np.log10(T),np.log10(T)+1)

def noise_covariance_estimation(y_new, y_old, L):
    Z = Mopi*(y_new-F*y_old)
    L = L*(k-1)/k + Z*Z/k
    return (L-CW)/S, L

def state_estimation(y_new, x_post, P_post, R_est):
    x_pre  = F*x_post
    P_pre  = F*P_post*F + Q
    K      = P_pre*H/(H*P_pre*H+R_est)
    x_post = x_pre + K*(y_new-H*x_pre)
    IKH    = 1 - K*H
    P_post = IKH*P_pre*IKH + K*R_est*K
    return x_post, P_post

print("State estimation using true noise covariance")
for i in range(N):
    for k in range(T):
        w, v = Q**0.5*npr.randn(), R**0.5*npr.randn()
        y_new = H*x + v
        x_post, P_post = state_estimation(y_new, x_post, P_post, R)
        x_post_history_true[i,k] = x_post
        x_true_history_true[i,k] = x
        x = F*x + w
        if k+1 in print_k:
            print('i = %3d, k = %8d: R_true = %8.6f, R_est = %8.6f' % (i+1, k+1, R, R))

print("State estimation using static certainty-equivalent noise covariance estimate")
for i in range(N):
    for k in range(T):
        w, v = Q**0.5*npr.randn(), R**0.5*npr.randn()
        y_new = H*x + v
        x_post, P_post = state_estimation(y_new, x_post, P_post, R_est)
        x_post_history_static[i,k] = x_post
        x_true_history_static[i,k] = x
        x = F*x + w
        if k+1 in print_k:
            print('i = %3d, k = %8d: R_true = %8.6f, R_est = %8.6f' % (i+1, k+1, R, R_est))

print("State estimation using adaptive noise covariance estimates")
for i in range(N):
    for k in range(T):
        w, v = Q**0.5*npr.randn(), R**0.5*npr.randn()
        y_new = H*x + v
        if k > 0:
            R_est_new, L = noise_covariance_estimation(y_new, y_old, L)
            if R_est_new > 0:
                R_est = R_est_new
        R_est_history[i,k] = R_est
        x_post, P_post = state_estimation(y_new, x_post, P_post, R_est)
        x_post_history_adaptive[i,k] = x_post
        x_true_history_adaptive[i,k] = x
        x = F*x + w
        y_old = np.copy(y_new)
        if k+1 in print_k:
            print('i = %3d, k = %8d: R_true = %8.6f, R_est = %8.6f' % (i+1, k+1, R, R_est))


percs = [0,25,50,75,100]

fig,ax = plt.subplots(figsize=(8, 6))
R_err = {}
for perc in percs:
    R_err[perc] = np.percentile(np.abs(R_est_history-R),perc,axis=0)
plt.loglog(k_hist, R_err[50], linewidth=2)
plt.fill_between(k_hist, R_err[25], R_err[75], facecolor=[0.5,0.5,0.5], alpha=0.5)
plt.fill_between(k_hist, R_err[0], R_err[100], facecolor=[0.8,0.8,0.8], alpha=0.5)
plt.ylim([R_err[25].min(),R_err[75].max()])
plt.xlabel("Time step (k)")
plt.ylabel("| R_est - R |")
plt.title("Estimation error of R vs time step")
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(20)
fig.subplots_adjust(left = 0.18,bottom = 0.18)

fig,ax = plt.subplots(figsize=(8, 6))
mse_true = np.cumsum(np.square(x_post_history_true-x_true_history_true),axis=1)/k_hist
mse_static = np.cumsum(np.square(x_post_history_static-x_true_history_static),axis=1)/k_hist
mse_adaptive = np.cumsum(np.square(x_post_history_adaptive-x_true_history_adaptive),axis=1)/k_hist
mse_true_perc = {}
mse_static_perc = {}
mse_adaptive_perc = {}
for perc in percs:
    mse_true_perc[perc] = np.percentile(mse_true,perc,axis=0)
    mse_static_perc[perc] = np.percentile(mse_static,perc,axis=0)
    mse_adaptive_perc[perc] = np.percentile(mse_adaptive,perc,axis=0)
plt.semilogx(k_hist, mse_true_perc[50], linewidth=2)
plt.semilogx(k_hist, mse_static_perc[50], linewidth=2)
plt.semilogx(k_hist, mse_adaptive_perc[50], linewidth=2)
plt.fill_between(k_hist, mse_true_perc[25], mse_true_perc[75], facecolor='tab:blue', alpha=0.20)
# plt.fill_between(k_hist, mse_true_perc[0], mse_true_perc[100], facecolor='tab:blue', alpha=0.05)
plt.fill_between(k_hist, mse_static_perc[25], mse_static_perc[75], facecolor='tab:orange', alpha=0.20)
# plt.fill_between(k_hist, mse_static_perc[0], mse_static_perc[100], facecolor='tab:orange', alpha=0.05)
plt.fill_between(k_hist, mse_adaptive_perc[25], mse_adaptive_perc[75], facecolor='tab:green', alpha=0.20)
# plt.fill_between(k_hist, mse_adaptive_perc[0], mse_adaptive_perc[100], facecolor='tab:green', alpha=0.05)
plt.xlabel("Time step (k)")
plt.ylabel("MSE")
plt.title("Mean square state estimation error vs time step")
plt.legend(["True","Static","Adaptive"])
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(20)
fig.subplots_adjust(left = 0.18,bottom = 0.18)

plt.show()
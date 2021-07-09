import numpy as np
import matplotlib.pyplot as plt

def u(t):
    return np.array([0, 0, 4+2*np.sin(1.2+2*t), 2+0.5*t, 0, 0])

def to_euler_angles(R):
    beta = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
    alpha = np.arctan2(R[1,0], R[0,0])
    gamma = np.arctan2(R[2,1], R[2,2])
    return np.array([alpha, beta, gamma])

def skew(n):
    S = np.zeros((3, 3))
    S[0, 1] = -n[2]
    S[0, 2] = n[1]
    S[1, 2] = -n[0]
    S[1, 0] = n[2]
    S[2, 0] = -n[1]
    S[2, 1] = n[0]
    return S

def angle_axis(n, theta):
    S = skew(n)
    return np.eye(3) + S*np.sin(theta) + np.matmul(S, S)*(1-np.cos(theta))

def screw_transform(l):
    theta = np.linalg.norm(l[0:3])
    T = np.eye(4)
    if theta==0:
        T[0:3, 3] = l[3:]
        return T
    n = l[0:3]/theta
    m_par = np.dot(l[3:], n)*n
    m_perp_hat = (l[3:] - m_par)/theta
    T[0:3, 0:3] = angle_axis(n, theta)
    T[0:3, 3] = m_par + m_perp_hat*np.sin(theta) + np.cross(n, m_perp_hat)*(1-np.cos(theta))
    return T

T = np.eye(4)

t0 = 0
t1 = 20
dt = 1e-2
n = int((t1-t0)/dt + 1)
t = np.linspace(t0, t1, n)

p = np.zeros((n, 3))
euler = np.zeros((n, 3))

for i in range(0, n):
    p[i, :] = T[0:3, 3]
    euler[i, :] = to_euler_angles(T[0:3, 0:3])
    if i==(n-1): break
    ti = t[i]
    dt = t[i+1] - t[i]
    dT = screw_transform(u(ti+dt/2)*dt)
    T = np.matmul(T, dT)

plt.plot(p[:, 0], p[:, 1])
plt.show()

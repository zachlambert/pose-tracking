# Provide odometry using a kinematic model

import numpy as np
import matplotlib.pyplot as plt

# Measured values and associated variance

def get_dq(t, dt):
    # Return the anglular displacement measured by each wheel
    # encoder.
    dqR = 1*np.exp(-5*t)
    dqL = 0.5*np.exp(-3*t)
    return np.array([dqR, dqL])

def get_dq_var(dq):
    return 0.2*np.diag(dq**2)

def get_J():
    b = 0.2
    r = 0.05
    J = np.array([
        [0, 0],
        [0, 0],
        [r/b, -r/b],
        [r/2, r/2],
        [0, 0],
        [0, 0]
    ])
    return J

# Function for integrating pose

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

def spatial_velocity_transform(T):
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    X = np.zeros((6, 6))
    X[0:3, 0:3] = R.transpose()
    X[3:, 0:3] = -R.transpose()*skew(p)
    X[3:, 3:] = R.transpose()
    return X

def get_p_var(T, T_var):
    R = T[0:3, 0:3]
    return np.matmul(R, np.matmul(T_var[3:, 3:], R.transpose()))

T = np.eye(4)
T_var = np.zeros((6, 6))

t0 = 0
t1 = 4
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
    # Update pose
    J = get_J()
    dq = get_dq(ti, dt)
    ds = np.matmul(J, dq)
    dT = screw_transform(ds)
    T = np.matmul(T, dT)
    # Update pose variance
    s_var = np.matmul(J, np.matmul(get_dq_var(dq), J.transpose()))
    dT_half = screw_transform(ds/2)
    A = spatial_velocity_transform(dT)
    B = spatial_velocity_transform(dT_half)
    T_var = np.matmul(A, np.matmul(T_var, A.transpose())) \
            + np.matmul(B, np.matmul(s_var, B.transpose()))

plt.plot(p[:, 0], p[:, 1])

# Plot a filled region for the p covariance
# Plots the 95% confidence region (1.96 standard deviations)
points = []
p_var = get_p_var(T, T_var)
print(p_var)
for angle in np.linspace(-np.pi, np.pi, 25):
    x = np.array([np.cos(angle), np.sin(angle), 0])
    dp = np.matmul(p_var, x)
    # Scale it with standard deviation, not variance
    dp = 1.96*np.sqrt(np.linalg.norm(dp)) * dp / np.linalg.norm(dp)
    points.append(p[-1,:] + dp)
points = np.array(points)
plt.fill(points[:,0], points[:, 1], "#aa88ff")

plt.show()

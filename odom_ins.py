# Provide odometry using an inertial navigation system

import numpy as np
import matplotlib.pyplot as plt

# Measured values and associated variance

def get_a(t, u):
    # The body experiences a centripetal acceleration and
    # a rate of change of body velocity (in body frame)
    # Choose acceleration here to correspond to a particular
    # rate of change of v
    v_dot = np.array([0.01, 0, 0])
    return v_dot + np.cross(u[0:3], u[3:])


def get_omega(t):
    return 0.1 * np.array([0, 0, 1])

def get_a_var(a):
    return 1e12 * np.diag(a**2)

def get_omega_var(omega):
    return 1e6 * np.diag(omega**2)

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
t1 = 20
dt = 1e-2
n = int((t1-t0)/dt + 1)
t = np.linspace(t0, t1, n)

p = np.zeros((n, 3))
euler = np.zeros((n, 3))

u = np.zeros(6)
u_dot = np.zeros(6)
u_var = np.zeros((6, 6))
for i in range(0, n):
    p[i, :] = T[0:3, 3]
    euler[i, :] = to_euler_angles(T[0:3, 0:3])
    if i==(n-1): break
    ti = t[i]
    dt = t[i+1] - t[i]
    # Update pose
    a = get_a(ti, u)
    u[0:3] = get_omega(ti)
    v_dot = a - np.cross(u[0:3], u[3:])
    u_dot[3:] = v_dot
    dT = screw_transform((u + u_dot*dt/2)*dt)
    T = np.matmul(T, dT)
    # Update pose variance
    dT_half = screw_transform((u + u_dot*dt/2)*dt/2)
    A = spatial_velocity_transform(dT)
    B = spatial_velocity_transform(dT_half) * (dt**2)
    T_var = np.matmul(A, np.matmul(T_var, A.transpose())) \
            + np.matmul(B, np.matmul(u_var, B.transpose()))
    # Update u variance (which uses current u)
    A = np.zeros((6, 6))
    A[3:, 0:3] = dt*skew(u[3:])
    A[3:, 3:] = np.eye(3) - dt*skew(u[0:3])
    B = np.zeros((6, 6))
    B[0:3, 0:3] = np.eye(3)
    B[3:, 3:] = np.eye(3) * (dt**2)
    in_var = np.zeros((6, 6))
    in_var[0:3, 0:3] = get_omega_var((u + u_dot*dt)[0:3])
    in_var[3:, 3:] = get_a_var(a)
    u_var = np.matmul(A, np.matmul(u_var, A.transpose())) \
            + np.matmul(B, np.matmul(in_var, B.transpose()))
    # Update u
    u += u_dot * dt

plt.plot(p[:, 0], p[:, 1])

# Plot a filled region for the p covariance
# Plot the 95% confidence region (1.96 standard deviations)
points = []
p_var = get_p_var(T, T_var)
print(p_var)
for angle in np.linspace(-np.pi, np.pi, 20):
    x = np.array([np.cos(angle), np.sin(angle), 0])
    dp = np.matmul(p_var, x)
    dp = 1.96*np.sqrt(np.linalg.norm(dp)) * dp / np.linalg.norm(dp)
    points.append(p[-1,:] + dp)
points = np.array(points)
plt.fill(points[:,0], points[:, 1], "#aa88ff")

plt.show()

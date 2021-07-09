import numpy as np
import matplotlib.pyplot as plt
import quaternion
from quaternion import (
    integrate_angular_velocity,
    as_rotation_matrix
)

def omega_body(t):
    return np.array([1*np.cos(t), 0.2, 1*np.sin(t)])
    # return np.array([0.3, 0.2, 1])

def omega_global(t, R):
    x = np.matmul(as_rotation_matrix(R), omega_body(t))
    return x

def omega_q(omega):
    return np.quaternion(0, *omega)

def as_euler_angles(q):
    R = as_rotation_matrix(q)
    beta = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
    alpha = np.arctan2(R[1,0], R[0,0])
    gamma = np.arctan2(R[2,1], R[2,2])
    return np.array([alpha, beta, gamma])

def as_euler_angles_array(q):
    euler = np.zeros((q.size, 3))
    for i, qi in enumerate(q):
        euler[i] = as_euler_angles(qi)
    return euler

def integrate_angular_velocity_euler(omega, t0, t1, q0, n):
    t = np.linspace(t0, t1, n)
    q = np.zeros(4*n, dtype=np.double).view(np.quaternion)
    qi = q0
    for i in range(n):
        q[i] = qi
        if i == n-1: break
        qi = qi*(1 + 0.5*(t[i+1]-t[i])*omega_q(omega_body(t[i])))
    return t, q

def integrate_angular_velocity_rk4(omega, t0, t1, q0, n):
    t = np.linspace(t0, t1, n)
    q = np.zeros(4*n, dtype=np.double).view(np.quaternion)
    qi = q0
    for i in range(0, n):
        q[i] = qi
        if i == n-1: break
        dt = t[i+1]-t[i]
        # k1 = 0.5*qi*omega_q(omega(t[i]))
        # k2 = 0.5*(qi+0.5*dt*k1)*omega_q(omega(t[i]+dt/2))
        # k3 = 0.5*(qi+0.5*dt*k2)*omega_q(omega(t[i]+dt/2))
        # k4 = 0.5*(qi+dt*k3)*omega_q(omega(t[i]+dt))
        # qi += (1/6)*dt*(k1 + 2*k2 + 2*k3 + k4)
        k1 = omega_q(omega(t[i]))
        k2 = (1+0.25*dt*k1)*omega_q(omega(t[i]+dt/2))
        k3 = (1+0.25*dt*k2)*omega_q(omega(t[i]+dt/2))
        k4 = (1+0.5*dt*k3)*omega_q(omega(t[i]+dt))
        qi = qi*(1 + (1/12)*dt*(k1 + 2*k2 + 2*k3 + k4))
    return t, q

def integrate_angular_velocity_exp(omega, t0, t1, q0, n):
    t = np.linspace(t0, t1, n)
    q = np.zeros(4*n, dtype=np.double).view(np.quaternion)
    qi = q0
    for i in range(0, n):
        q[i] = qi
        if i == n-1: break
        dt = t[i+1]-t[i]
        exp_coords = omega(t[i]+dt/2)*dt
        theta = np.linalg.norm(exp_coords)
        axis = exp_coords / theta
        dq = np.quaternion(np.cos(theta/2), *axis*np.sin(theta/2))
        qi = qi*dq
    return t, q

q0 = np.quaternion(1, 0, 0, 0)
t0 = 0
t1 = 20
dt = 1e-1 # Update every 100ms
n = int((t1 - t0)/dt + 1)

t, q = integrate_angular_velocity(omega_global, t0, t1, q0, 1e-20)
euler = as_euler_angles_array(q)
t2, q2 = integrate_angular_velocity_euler(omega_body, t0, t1, q0, n)
euler2 = as_euler_angles_array(q2)
t3, q3 = integrate_angular_velocity_rk4(omega_body, t0, t1, q0, n)
euler3 = as_euler_angles_array(q3)
t4, q4 = integrate_angular_velocity_exp(omega_body, t0, t1, q0, n)
euler4 = as_euler_angles_array(q4)

plt.plot(t, euler[:, 0], label="roll_accurate")
plt.plot(t, euler[:, 1], label="pitch_accurate")
plt.plot(t, euler[:, 2], label="yaw_accurate")
# plt.plot(t2, euler2[:, 0], label="roll_euler")
# plt.plot(t2, euler2[:, 1], label="pitch_euler")
# plt.plot(t2, euler2[:, 2], label="yaw_euler")
plt.plot(t3, euler3[:, 0], label="roll_rk4")
plt.plot(t3, euler3[:, 1], label="pitch_rk4")
plt.plot(t3, euler3[:, 2], label="yaw_rk4")
plt.plot(t4, euler4[:, 0], label="roll_exp")
plt.plot(t4, euler4[:, 1], label="pitch_exp")
plt.plot(t4, euler4[:, 2], label="yaw_exp")
plt.legend()
plt.show()

# Even updating slowly at 100ms, still remains pretty accurate,
# even for euler.

# Using at higher rate (eg: every 10ms) will give decent accuracy
# with a simple euler method.

# Using RK4 with the higher rate will give even more accuracy, and
# is worth using when you need that accuracy.

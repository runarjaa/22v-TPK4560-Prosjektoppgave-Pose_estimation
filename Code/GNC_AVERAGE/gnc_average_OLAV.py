import numpy as np
from sympy import maximum

def w_from_r(r, eps, mu):
    if abs(r) < eps * np.sqrt(mu/(mu+1)):
        w = 1
    elif abs(r) > eps * np.sqrt((mu+1)/mu):
        w = 0
    else:
        w = (eps * np.sqrt(mu*(mu+1)))/abs(r) - mu
    return w

# Preparations
np.set_printoptions(formatter={"float": "{: 0.4f}".format})

# Generation of input data
y = np.array([2.0, 2.1, 1.9, 1.95, 2.07, 1.93, 2.01, 5, 6, -5, -6, 100,101])
n = y.shape[0]

# initial value
x0 = 1.0
# Loss function intial data
r = np.zeros(n)
for i in range(n):
    r[i] = np.linalg.norm(y[i] - x0)
r_0_max = np.max(r)
print("\n r0 = \n {}\n".format(r))

# GNC initialization
eps = 0.2; mu_update_factor = 1.4
max_iterations = 1000
w = np.ones(n)
mu = eps**2 / (2*r_0_max**2 - eps**2)
print("\n mu = \n {}\n".format(mu))

for i in range(max_iterations):
    # Weighted average
    x = np.dot(y,w)/np.sum(w)

    # Loss function
    for j in range(n):
        r[j] = np.linalg.norm(y[j] - x)
        w[j] = w_from_r(r[j], eps, mu)
    
    mu = mu_update_factor * mu

print("\n x = \n {}\n".format(x))
print("\n r = \n {}\n".format(r))
print("\n w = \n {}\n".format(w))
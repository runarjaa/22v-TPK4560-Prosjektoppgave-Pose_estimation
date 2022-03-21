import numpy as np

def skewm(r):
    return np.array([[0,-r[2],r[1]], [r[2],0,-r[0]], [-r[1],r[0],0]])

def expso3(u):
    S = skewm(u); un = np.linalg.norm(u)
    return np.eye(3) + np.sinc(un/np.pi)*S \
        + 0.5*(np.sinc(un/(2*np.pi)))**2 * S@S

def w_from_r(r, eps, mu):
    if abs(r) < eps * np.sqrt(mu/(mu+1)):
        w = 1
    elif abs(r) > eps * np.sqrt((mu+1)/mu):
        w = 0
    else:
        w = (eps * np.sqrt(mu*(mu+1)))/abs(r) - mu
    return w

# Prep
np.set_printoptions(formatter={"float": "{: 0.4f}".format})
ex = np.array([1.,0.,0.])
ey = np.array([0.,1.,0.])
ez = np.array([0.,0.,1.])

# Generation of input data
B = np.block([[ex], [ex], [ex], [ex], [ex], [ey], [ex]]).T
n = B.shape[1]
A = np.zeros((3, n))
Ra = np.stack((expso3(np.pi/6*ez),
                expso3((np.pi/6*ez - 0.01*ex)),
                expso3((np.pi/6*ez + 0.01*ex)),
                expso3((np.pi/6*ez - 0.1*ex)),
                expso3((np.pi/6*ez + 0.1*ex)),
                expso3((np.pi/6 + 0.5*np.pi/2)*ez),
                expso3((np.pi/6 + 0.5*np.pi/2)*ey)),
                axis = 0)

for i in range(n):
    A[:,i] = Ra[i]@B[:,i]
R0 = np.identity(3)

# Loss function intial data
r = np.zeros(n)
for i in range(n):
    r[i] = np.linalg.norm(A[:,i] - R0@B[:,i])
r_0_max = np.max(r)
print("\n r0 = {}\nr_0_max = {}\n".format(r, r_0_max))

# GNC initialization
eps = 0.011; mu_update_factor = 1.4
max_iterations = 1000
w = np.ones(n)
mu = eps**2 / (2*r_0_max**2 - eps**2)
print("\n mu = \n {}\n".format(mu))

for i in range(max_iterations):
    # Weighted Procrustes
    H = B @ np.diag(w) @ A.T
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ np.diag([1,1,np.linalg.det( Vt.T @ U.T)]) @ U.T

    # Loss function
    for j in range(n):
        r[j] = np.linalg.norm(A[:,j] - R@B[:,j])
        w[j] = w_from_r(r[j], eps, mu)

    mu = mu_update_factor * mu

print("\n R = \n {}\n".format(R))
print("\n r = \n {}\n".format(r))
print("\n w = \n {}\n".format(w))
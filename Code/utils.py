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

def expso3(u):
    R = np.identity(3) + np.sinc(np.linalg.norm(u)/np.pi)*skewm(u)\
        + 0.5*(np.sinc(np.linalg.norm(u)/(2*np.pi)))**2 * skewm(u) @ skewm(u)
    return R
import numpy as np
import matplotlib.pyplot as plt
import math

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


# Random data
focal = 800
u_0 = 640
v_0 = 512

# Transformation Matrix
transf_c_w = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1,-2],
    [0, 0, 0, 1]
])

# Parameter matrix
camera_para =  np.array([
    [focal,     0, u_0],
    [    0, focal, v_0],
    [    0,     0,   1]
])

pi_mat = np.eye(3,4)

# Define Control Points in World coor coor - control world
ch_w = np.array([
    [0, 0, 0, 1],
    [1, 0, 0, 1], 
    [0, 1, 0, 1], 
    [0, 0, 1, 1]
]).T

# Points in world coordinates - Homogenous
ph_w = np.array([
    [ 0.995555881938209 ,  0.4802387824086085 ,  0.01206817493434742 , 1.],
    [ 0.48201950525607407 ,  0.03667711951818575 ,  0.2247198631984184 , 1.],
    [ 0.39687422295226926 ,  0.4468422492615619 ,  0.5579296060298905 , 1.],
    [ 0.1307361548312025 ,  0.0017949235578693656 ,  0.7992104944714973 , 1.],
    [ 0.1802038827293625 ,  0.30147195450672637 ,  0.3065432070634452 , 1.],
    [ 0.1273136116768765 ,  0.6008489312233133 ,  0.814072408443261 , 1.],
    [ 0.5936937283176116 ,  0.5617129604604256 ,  0.21464775634992062 , 1.],
    [ 0.07119734746580442 ,  0.777510858755277 ,  0.7501307315179284 , 1.],
    [ 0.32389704460643065 ,  0.8900472338710688 ,  0.15140266567649263 , 1.],
    [ 0.4290695543893176 ,  0.4611765374631873 ,  0.9408351719872696 , 1.]
]).T
n = np.shape(ph_w)[1]
ph_w = ph_w.reshape(4,n)

# Points in camera coordinates - Homogenous
ph_c = np.dot(transf_c_w, ph_w).reshape(4,n)

# Cartesian(?) points
# Dont think this is even neccesary
p_w = np.dot(np.eye(3,4), ph_w).reshape(3,n)
p_c = np.dot(np.eye(3,4), ph_c).reshape(3,n)

# Normalized image coordinates
s_norm = np.empty_like(p_c)
for i in range(n):
    temp = (1/p_c[2,i])
    s_norm[0,i] = p_c[0,i]*temp
    s_norm[1,i] = p_c[1,i]*temp
    s_norm[2,i] = p_c[2,i]*temp

# Pixel coordinates
p_hat = np.dot(camera_para, s_norm)


pix_yo = camera_para @ pi_mat @ transf_c_w @ ph_w

for i in range(n):
    temp = (1/pix_yo[2,i])
    pix_yo[0,i] = np.rint(pix_yo[0,i]*temp)
    pix_yo[1,i] = np.rint(pix_yo[1,i]*temp)
    pix_yo[2,i] = np.rint(pix_yo[2,i]*temp)


# Cpmpute alphas
alpha = np.matmul(np.linalg.inv(ch_w),ph_w).T

# Calculate M matrix
M = []
for i in range(n):
    M.append([
        alpha[i, 0] * focal, 0, alpha[i, 0] * (u_0 - pix_yo[i, 0]),
        alpha[i, 1] * focal, 0, alpha[i, 1] * (u_0 - pix_yo[i, 0]),
        alpha[i, 2] * focal, 0, alpha[i, 2] * (u_0 - pix_yo[i, 0]),
        alpha[i, 3] * focal, 0, alpha[i, 3] * (u_0 - pix_yo[i, 0])
    ])
    M.append([
        0, alpha[i, 0] * focal, alpha[i, 0] * (u_0 - pix_yo[i, 1]),
        0, alpha[i, 1] * focal, alpha[i, 1] * (u_0 - pix_yo[i, 1]),
        0, alpha[i, 2] * focal, alpha[i, 2] * (u_0 - pix_yo[i, 1]),
        0, alpha[i, 3] * focal, alpha[i, 3] * (u_0 - pix_yo[i, 1])
    ])
print(M)




#Plotting points
# fig_1 = plt.figure()
# fig_2 = plt.figure()
# ax = fig_1.add_subplot(projection= '3d')
# ax.set_xlim(0,u_0*2)
# ax.set_ylim(0,v_0*2)
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')
# ay = fig_2.add_subplot(projection= '3d')
# ay.set_ylabel('Y-axis')
# ay.set_zlabel('Z-axis')
# ay.set_xlabel('X-axis')

# for i in range(n):
#     ax.scatter(pix_yo[0][i], pix_yo[1][i], pix_yo[2][i])
#     ay.scatter(p_hat[0][i], p_hat[1][i], p_hat[2][i])
# plt.show()


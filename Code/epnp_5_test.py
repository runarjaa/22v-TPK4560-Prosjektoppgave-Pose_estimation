import numpy as np
import matplotlib.pyplot as plt
import random as rand
from datetime import datetime
from epnp_5 import EPnP


rand.seed(datetime.now())
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

# Trying to import stanford bunny
bunny = np.loadtxt("d:/Skole/Semester 10/Prosjektoppgave/Code_Project_2022_Jåtun/Code/bunny_points.txt", dtype=np.float64)
bunnyh = np.hstack((bunny, np.ones((bunny.shape[0],1))))

# Random data
focal = 1500
u_0 = 640
v_0 = 512

# # Transformation Matrix
transf_c_w = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1  ]
], dtype=np.float64)

# # Parameter matrix
camera_para =  np.array([
    [focal,     0, u_0],
    [    0, focal, v_0],
    [    0,     0,   1]
])

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
    ])


if __name__ == "__main__":
    epnp = EPnP()
    # epnp.load_random_data(100,60,60,0,0,0,3,1500,1500,640,512)
    epnp.load_set_data(transf_c_w, camera_para, bunnyh)
    epnp.compute_reg_epnp()
    # print(epnp.Rt_3)
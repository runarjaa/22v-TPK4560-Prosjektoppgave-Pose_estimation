import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import open3d as o3d
import random as rand
from datetime import datetime
rand.seed(datetime.now())

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

class EPnP:
    def __init__(self) -> None:
        self.ch_w = self.control_points()

    def compute_reg_epnp(self):
        self.alpha = self.compute_alpha()
        self.M = self.compute_M()
        self.K = self.compute_K()
        self.rho = self.compute_rho()
        self.L_6_10 = self.compute_L_6_10()

        self.betas = self.compute_betas()
        self.X1, self.X2, self.X3 = self.compute_Xi()
        self.c_c1, self.x_c1, self.sc_1 = self.compute_norm_sign_scaling_factor(self.X1, self.x_w)
        self.c_c2, self.x_c2, self.sc_2 = self.compute_norm_sign_scaling_factor(self.X2, self.x_w)
        self.c_c3, self.x_c3, self.sc_3 = self.compute_norm_sign_scaling_factor(self.X3, self.x_w)
        self.Rt_1 = self.getRotT(self.x_w, self.x_c1)
        self.Rt_2 = self.getRotT(self.x_w, self.x_c2)
        self.Rt_3 = self.getRotT(self.x_w, self.x_c3)
        self.best_trans()
        
        self.compute_pixels()
        # self.check_opencv()
        self.print_results()
    
    # def compute_gnc_tls_epnp(self):
        

    # def loss(self, a, T, b):
    #     return a - self.C @ T @ b.T

    
    def load_random_data(self, n, ax, ay, az, x, y, z, fu, fv, u0, v0, randomT=False):
        self.n = n
        self.fu = fu
        self.fv = fv
        self.u0 = u0
        self.v0 = v0

        # Transformation matrix and camera matrix
        if randomT == False:
            self.T = self.compute_T(ax, ay, az, x, y, z)
        else:
            self.T = self.compute_T(rand.random(), rand.random(), rand.random(), rand.random(), rand.random(), rand.random())
        
        if randomT == False:
            self.C = self.compute_C(fu, fv, u0, v0)
        else:
            self.C = self.compute_C(rand.random(), rand.random(), rand.random(), rand.random())
        
        # Points and stuff
        self.xh_w = np.empty((self.n,4))
        for i in range(self.n):
            self.xh_w[i,:] = np.array([rand.random(), rand.random(), rand.random(), 1.0])
        
        self.compute_points()
        

    def load_set_data(self, Tr, Ca, points_w):
        self.n = points_w.shape[0]
        self.fu = Ca[0,0]
        self.fv = Ca[1,1]
        self.u0 = Ca[0,2]
        self.v0 = Ca[1,2]
        
        self.T = Tr
        self.C = Ca
        self.xh_w = points_w
        self.compute_points()


    # Calculate points based on T, C and ph_w
    def compute_points(self):

        # Reference points not homogenous
        self.x_w = (np.eye(3,4) @ self.xh_w.T).T
        self.x_c_actual = (np.eye(3,4) @ self.T @ self.xh_w.T).T
        # Reference points as pixels
        pix_true = (self.C @ np.eye(3,4) @ self.T @ self.xh_w.T).T
        self.pix_true = np.rint(pix_true*(1/(pix_true[:,2]).reshape((self.n,1))))
        
        # Reference points with some random noise to be used for calculation
        self.pix = self.pix_true.copy()
        for i, p in enumerate(self.pix):
            if i % 10 == 0:
                p[0] += rand.randint(-10,10) 
                p[1] += rand.randint(-10,10) 
        
        # Reference points as normalized coordinates
        self.snorm =  (self.T @ self.xh_w.T).T

    # Computing alphas
    def compute_alpha(self):
        X = self.xh_w.T
        C = self.ch_w.T
        return (np.matmul(np.linalg.inv(C), X).T)

    # Computing M matrix from Mx = 0
    def compute_M(self):
        M = np.empty((2*self.n, 12))
        for i in range(self.n):
            M[i*2,:]= [
                self.alpha[i, 0] * self.fu, 0, self.alpha[i, 0] * (self.u0 - self.pix[i, 0]),
                self.alpha[i, 1] * self.fu, 0, self.alpha[i, 1] * (self.u0 - self.pix[i, 0]),
                self.alpha[i, 2] * self.fu, 0, self.alpha[i, 2] * (self.u0 - self.pix[i, 0]),
                self.alpha[i, 3] * self.fu, 0, self.alpha[i, 3] * (self.u0 - self.pix[i, 0])
            ]
            M[i*2+1,:] = [
                0, self.alpha[i, 0] * self.fv, self.alpha[i, 0] * (self.v0 - self.pix[i, 1]),
                0, self.alpha[i, 1] * self.fv, self.alpha[i, 1] * (self.v0 - self.pix[i, 1]),
                0, self.alpha[i, 2] * self.fv, self.alpha[i, 2] * (self.v0 - self.pix[i, 1]),
                0, self.alpha[i, 3] * self.fv, self.alpha[i, 3] * (self.v0 - self.pix[i, 1])
            ]
        return M
    
    # Computing eighenvalues and eigenvectors of MtM
    def compute_K(self):
        M = self.M
        MtM = np.matmul(M.T, M)
        eig_val, eig_vec = np.linalg.eig(MtM)
        sort = eig_val.argsort()
        return eig_vec[:,sort[:4]]

    # Computing rho from Lb = p
    def compute_rho(self):
        return np.array([
            np.linalg.norm(self.ch_w[:,0]-self.ch_w[:,1])**2,
            np.linalg.norm(self.ch_w[:,0]-self.ch_w[:,2])**2,
            np.linalg.norm(self.ch_w[:,0]-self.ch_w[:,3])**2,
            np.linalg.norm(self.ch_w[:,1]-self.ch_w[:,2])**2,
            np.linalg.norm(self.ch_w[:,1]-self.ch_w[:,3])**2,
            np.linalg.norm(self.ch_w[:,2]-self.ch_w[:,3])**2
        ])

# -----------------------------------------------------------------------------
 # -----------------------------------------------------------------------------
  # -----------------------------------------------------------------------------
   # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # Computing L matrix from Lb = p
    # Did not understand this, so this is borrowed/stolen from EPnP python
    def compute_L_6_10(self):
        L_6_10 = np.zeros((6,10))
        kernel = np.array([self.K.T[3], self.K.T[2], self.K.T[1], self.K.T[0]]).T

        v = []
        for i in range(4):
            v.append(kernel[:, i])

        dv = []

        for r in range(4):
            dv.append([])
            for i in range(3):
                for j in range(i+1, 4):
                    dv[r].append(v[r][3*i:3*(i+1)]-v[r][3*j:3*(j+1)])

        index = [
            (0, 0),
            (0, 1),
            (1, 1),
            (0, 2),
            (1, 2),
            (2, 2),
            (0, 3),
            (1, 3),
            (2, 3),
            (3, 3)
            ]

        for i in range(6):
            j = 0
            for a, b in index:
                L_6_10[i, j] = np.matmul(dv[a][i], dv[b][i].T)
                if a != b:
                    L_6_10[i, j] *= 2
                j += 1

        return L_6_10
    
    # Calculating lesser matrices - Also stolen
    def compute_L_6_6(self):
        return self.L_6_10[:, (2, 4, 7, 5, 8, 9)]
    def compute_L_6_3(self):
        return self.L_6_10[:, (5,8,9)]

    
    # Calculating scaling factor along with vectors
    # He can't keep getting away with stealing like this
    def compute_norm_sign_scaling_factor(self, X, Xworld):
        Cc = []
    
        for i in range(4):
            Cc.append(X[(3 * i) : (3 * (i + 1))])
        
        Cc = np.array(Cc).reshape((4, 3))

        Xc = np.matmul(self.alpha, Cc)
        
        centr_w = np.mean(Xworld, axis=0)
        centroid_w = np.tile(centr_w.reshape((1, 3)), (self.n, 1))
        tmp1 = Xworld.reshape((self.n, 3)) - centroid_w
        dist_w = np.sqrt(np.sum(tmp1 ** 2, axis=1))
        
        centr_c = np.mean(np.array(Xc), axis=0)
        centroid_c = np.tile(centr_c.reshape((1, 3)), (self.n, 1))
        tmp2 = Xc.reshape((self.n, 3)) - centroid_c
        dist_c = np.sqrt(np.sum(tmp2 ** 2, axis=1))
        
        sc_1 = np.matmul(dist_c.transpose(), dist_c) ** -1
        sc_2 = np.matmul(dist_c.transpose(), dist_w)
        sc = sc_1 * sc_2
        
        Cc *= sc
        Xc = np.matmul(self.alpha, Cc)
        
        for x in Xc:
            if x[-1] < 0:
                Xc *= -1
                Cc *= -1
        
        return Cc, Xc, sc

    # #stolenagain - Turns out it is just Procrustes
    def getRotT(self, wpts, cpts):
        wcent = np.tile(np.mean(wpts, axis=0).reshape((1, 3)), (self.n, 1))
        ccent = np.tile(np.mean(cpts, axis=0).reshape((1, 3)), (self.n, 1))
        wpts = (wpts.reshape((self.n, 3)) - wcent)
        cpts = (cpts.reshape((self.n, 3)) - ccent)
        
        M = np.matmul(cpts.transpose(), wpts)
        
        U, S, V = np.linalg.svd(M)
        R = np.matmul(U, V)
        
        if np.linalg.det(R) < 0:
            R = - R
        T = ccent[0].transpose() - np.matmul(R, wcent[0].transpose())

        Rt = np.concatenate((R.reshape((3, 3)), T.reshape((3, 1))), axis=1)
        
        return Rt
# -----------------------------------------------------------------------------
 # -----------------------------------------------------------------------------
  # -----------------------------------------------------------------------------
   # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

 
    # Computing betas
    def compute_betas(self):
        L_6_6 = self.compute_L_6_6()
        L_6_3 = self.compute_L_6_3()

        beta = np.zeros((3,3))
        beta[0,0] = 1

        temp = np.matmul(np.linalg.pinv(L_6_3), self.rho)
        beta[1,0] = np.sqrt(abs(temp[0]))
        beta[1,1] = np.sqrt(abs(temp[1]))

        temp = np.matmul(np.linalg.inv(L_6_6), self.rho)
        beta[2,0] = np.sqrt(abs(temp[0]))
        beta[2,1] = np.sqrt(abs(temp[3]))
        beta[2,2] = np.sqrt(abs(temp[5]))

        return beta

    # Computing x in solution for x = sum(beta*v) for cas 1->3
    def compute_Xi(self):
        X1 = self.betas[0,0] * self.K[:, 0]
        X2 = self.betas[1,0] * self.K[:, 1] + self.betas[1,1]*self.K[:, 0]
        X3 = self.betas[2,0] * self.K[:, 2] + self.betas[2,1]*self.K[:, 1] + self.betas[2,2]*self.K[:, 0]
        return X1, X2, X3


    def best_trans(self):
        err1 = np.linalg.norm(self.T[:3,:] - self.Rt_1)
        err2 = np.linalg.norm(self.T[:3,:] - self.Rt_2)
        err3 = np.linalg.norm(self.T[:3,:] - self.Rt_3)
        if err1 < err2 and err1 < err3:
            self.Rt_best = self.Rt_1
            self.err_best = err1
            self.best_rot_idx = 1
        elif err2 < err1 and err2 < err3:
            self.Rt_best = self.Rt_2
            self.err_best = err2
            self.best_rot_idx = 2
        elif err3 < err1 and err3 < err2:
            self.Rt_best = self.Rt_3
            self.err_best = err3
            self.best_rot_idx = 3
        else:
            print("Could not find best rotation matrix")

    # Defning control points
    def control_points(self):
        return  np.array([
                [1, 0, 0, 1], 
                [0, 1, 0, 1], 
                [0, 0, 1, 1],
                [0, 0, 0, 1]
            ], dtype=float).T
    
    # Transformation matrix
    def compute_Rx(self, angle):
        angle = angle*(np.pi/180)
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle),  np.cos(angle)]
        ])
    def compute_Ry(self, angle):
        angle = angle*(np.pi/180)
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0 ],
            [-np.sin(angle), 0,  np.cos(angle)]
        ])
    def compute_Rz(self, angle):
        angle = angle*(np.pi/180)
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle),  np.cos(angle)]
        ])

    def compute_Tr(self, x,y,z):
        return np.array([x,y,z])

    def compute_T(self, anglex, angley, anglez, x, y, z):
        R = self.compute_Rz(anglez) @ self.compute_Ry(angley) @ self.compute_Rx(anglex)
        T = self.compute_Tr(x,y,z)
        temp1 = np.concatenate((R.reshape(3,3), T.reshape((3,1))), axis=1)
        temp2 = np.concatenate((temp1, np.array([[0,0,0,1]]).reshape((1,4))))
        return temp2
    
    # Camera matrix
    def compute_C(self, fu, fv, u0, v0):
        return np.array([
            [  fu,   0, u0],
            [    0, fv, v0],
            [    0,  0,  1]
        ])
    
    # Looking at OpenCV
    def check_opencv(self):
        x_w = (self.x_w)#.reshape((self.n, 3, 1))
        pix = (self.pix[:,:2])#.reshape((self.n, 2, 1))
        C = self.C
        success_cv, rotation_cv, trans_cv = cv.solvePnP(x_w, pix, C, None)
        transf_cv = np.concatenate((rotation_cv.reshape((3, 3)), trans_cv.reshape((3, 1))), axis=1)
        self.x_c_cv = (np.eye((3,4)) @ trans_cv @ self.xh_w.T).T
        snorm_cv = self.x_c_cv*(1/self.x_c_cv[:,2].reshape((self.n,1)))
        self.pix_cv = snorm_cv @ self.C.T

    def compute_pixels(self):
        if self.best_rot_idx == 1:
            snorm_1 = self.x_c1*(1/self.x_c1[:,2].reshape((self.n,1)))
            self.pix_calc = np.rint(snorm_1 @ self.C.T)
        elif self.best_rot_idx == 2:
            snorm_2 = self.x_c2*(1/self.x_c2[:,2].reshape((self.n,1)))
            self.pix_calc = np.rint(snorm_2 @ self.C.T)
        elif self.best_rot_idx == 3:
            snorm_3 = self.x_c3*(1/self.x_c3[:,2].reshape((self.n,1)))
            self.pix_calc = np.rint(snorm_3 @ self.C.T)

    # Printing 3d image
    def plot_results_plt(self):
        fig_1 = plt.figure()
        ax = fig_1.add_subplot(projection='3d')
        # ax.set_xlim(-1,1)
        # ax.set_ylim(-1,1)
        # ax.set_zlim(-3,3)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        # ax.scatter(0,0,0, c='purple')

        for i in range(self.n):
            ax.scatter( self.x_c_actual[i][0]   ,  self.x_c_actual[i][1]    ,  self.x_c_actual[i][2], c='blue')
            ax.scatter( self.x_c1[i][0]         ,  self.x_c1[i][1]          ,  self.x_c1[i][2]      , c='red')
            # ax.scatter( self.x_c2[i][0]         ,  self.x_c2[i][1]          ,  self.x_c2[i][2]      , c='green')
            # ax.scatter( self.x_c3[i][0]         ,  self.x_c3[i][1]          ,  self.x_c3[i][2]      , c='cyan')

            # ax.scatter(self.x_c_cv[i][0], self.x_c_cv[i][1], self.x_c_cv[i][2], c='orange')
        plt.show()

        fig_2 = plt.figure()
        ay = fig_2.add_subplot()
        # ay.set_xlim(0,self.u0*2)
        # ay.set_ylim(0,self.v0*2)
        ay.set_xlabel('X-axis')
        ay.set_ylabel('Y-axis')

        for i in range(self.n):
            ay.scatter(self.pix_true[i][0], self.pix_true[i][1], c="blue", marker='s')
            ay.scatter(self.pix_1[i,0], self.pix_1[i,1], c="red", marker='o')
            # ay.scatter(self.pix_2[i,0], self.pix_2[i,1], c="green", marker='>')
            # ax.scatter(self.pix_3[i,0], self.pix_3[i,1], c="cyan", marker='o')

            # ay.scatter(self.pix_cv[i,0], self.pix_cv[i,1], c="orange", marker='>')
        plt.show()
    
    def plot_results_o3d(self):
        # Colors to be used
        color1 = np.array([0.0, 0.0, 1.0])
        color2 = np.array([255,165,0])/255
        # Actual pixels
        pcd_true = o3d.geometry.PointCloud()
        pcd_true.points = o3d.utility.Vector3dVector(self.pix_true)
        pcd_true.paint_uniform_color(color1)
        # Calculated Pixeld
        pcd_epnp = o3d.geometry.PointCloud()
        pcd_epnp.points = o3d.utility.Vector3dVector(self.pix_calc)
        pcd_epnp.paint_uniform_color(color2)
        # Drawing the pixels
        o3d.visualization.draw_geometries([pcd_true, pcd_epnp], width=1600, height=900)



    def print_results(self):
        print("Results:")
        print("Actual Transfomration matrix:")
        print(self.T[:3,:])
        print("\nBest calculated Transfomation matrix:")
        print(self.Rt_best)
        print("Beta:", self.best_rot_idx)
        print("Error:", self.err_best)

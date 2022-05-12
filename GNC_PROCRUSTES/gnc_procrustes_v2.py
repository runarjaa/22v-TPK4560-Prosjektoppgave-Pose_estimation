# from tkinter import XView
from utils import *

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

class GNC_PROCRUSTES:
    def __init__(self, outlier_percentage = 0) -> None:
        self.o_p = outlier_percentage
        self.geometry_list = []
        self.load_points()
        self.gnc()
        pass

    def load_points(self):
        # Loading ply-file
        path_to_bunny = "D:\\Skole\\Semester_10\\Prosjektoppgave\\Data\\bunny\\reconstruction\\bun_zipper_res3.ply"
        bunny_true_ply = o3d.io.read_point_cloud(path_to_bunny)

        # Point calculations
        self.bunny = np.asarray(bunny_true_ply.points)
        np.random.shuffle(self.bunny)
        self.bunny_true = np.copy(self.bunny)
        self.n = self.bunny.shape[0]
        self.o_i = np.rint(self.n * (self.o_p/100)).astype(np.int32)

        self.rot = expso3(np.array([np.pi * np.random.uniform(-1,1),
                    np.pi * np.random.uniform(-1,1),
                    np.pi * np.random.uniform(-1,1)
                    ]))
        self.bunny = self.bunny @ self.rot

        for i in range(self.o_i):
            self.bunny[i] = self.outlier_rot() @ self.bunny[i]

    def gnc(self):
        # Initialization
        R0 = np.identity(3)
        self.r = np.zeros(self.n)
        for i in range(self.n):
            self.r[i] = np.linalg.norm(self.bunny_true[i] - R0 @ self.bunny[i])**2
        r0_max = np.max(self.r)

        eps = 0.011
        mu_update = 1.4
        max_iter = 1000
        
        self.w = np.ones(self.n)
        mu = eps**2 / (2*r0_max**2 - eps**2)

        # Iteration
        last_iter = []
        self.iterations = 0
        for i in range(max_iter):
            self.iterations += 1
            last_iter.append(np.sum(self.w))
            # Weighted Procrustes
            H = self.bunny.T @ np.diag(self.w) @ self.bunny_true
            U, S, Vt = np.linalg.svd(H)
            self.R = Vt.T @ np.diag([1,1,np.linalg.det( Vt.T @ U.T)]) @ U.T

            # Loss function
            for j in range(self.n):
                self.r[j] = np.linalg.norm(self.bunny_true[j] - self.R @ self.bunny[j])**2
                self.w[j] = w_from_r(self.r[j], eps, mu)
            
            mu = mu_update * mu

            if i >= 5:
                if np.sum(self.w) == last_iter[i]:
                    break



# -------------- Calculations --------------
    def calculate_perc(self):
        inl = []
        out = []

        for i, n in enumerate(self.w):
            if self.w[i] == 1.0:
                inl.append(n)
            else:
                out.append(n)
        
        self.inliers = len(inl)
        self.outliers= len(out)

        self.percentage = (1-self.inliers/(self.outliers + self.inliers))*100

        # print("Inliers:\t", self.inliers, "\nOutliers:\t", self.outliers,
        #     "\nPercentage:\t",self.percentage, 
        #     "%\n\nIterations:\t", self.iterations)


    def important_info(self):
        """
        Returns important info for plotting and printing
        self.r              loss 
        self.w              weight
        self.inliers        number of inliers
        self.outliers       number of outliers
        self.percentage     percentage outliers vs inliers

        """
        return 
# ------------------------------------------    


# ---------------- Utilities ---------------
    def outlier_rot(self):
        outlier = np.array([np.pi * np.random.uniform(-1,1),
                    np.pi * np.random.uniform(-1,1),
                    np.pi * np.random.uniform(-1,1)
                    ])
        return expso3(outlier)
# ------------------------------------------

# ---------------- Plotting ----------------
    def make_ply(self):
        pcd_true = o3d.geometry.PointCloud()
        pcd_true.points = o3d.utility.Vector3dVector(self.bunny_true)

        pcd_bunny = o3d.geometry.PointCloud()
        pcd_bunny.points = o3d.utility.Vector3dVector(self.bunny)

        pcd_true_rot = o3d.geometry.PointCloud()
        pcd_true_rot.points = o3d.utility.Vector3dVector(
            self.bunny_true @ self.rot + np.ones((self.bunny_true.shape[0],3))*0.001)
        
        pcd_bunny_cal = o3d.geometry.PointCloud()
        pcd_bunny_cal.points = o3d.utility.Vector3dVector(
            self.bunny_true @ self.R)

        # Choose the point clouds to be visualised

        # self.geometry_list.append(pcd_true)
        self.geometry_list.append(pcd_bunny)
        # self.geometry_list.append(pcd_true_rot)
        self.geometry_list.append(pcd_bunny_cal)


    def plot_o3d(self):
        self.make_ply()


        n = len(self.geometry_list)
        for i in range(n):
            color = np.array([1-i/n, i/n, 0.0])
            self.geometry_list[i].paint_uniform_color(color)
        
        plot_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05, origin=[0, 0, 0])
        self.geometry_list.append(plot_axis)
        o3d.visualization.draw_geometries(self.geometry_list, width=1600, height=900)
        o3d.visualization.draw_geometries([self.geometry_list[0], self.geometry_list[2]], width=1600, height=900)
# ------------------------------------------



# ----------------- Testing ----------------
def testing_gnc_procrustes(
            min_percentage = 0, 
            max_percentage = 90, 
            step = 10,
            num_per_percent = 1
        ):
    
    print("Testing Procrustes with GNC")

    average_error_all = []
    average_iterations_all = []
    accuracy_percentage_all = []


    for i in range(min_percentage, max_percentage+1, step):
        average_error = []
        average_iterations = []
        accuracy_percent = []
        for j in range (num_per_percent):
            gnc = GNC_PROCRUSTES(outlier_percentage=i)

            gnc.calculate_perc()
            error = np.linalg.norm(gnc.rot  - gnc.R )
            # error = np.linalg.norm(gnc.rot @ gnc.bunny_true.T - gnc.R @ gnc.bunny_true.T)
            average_error.append(error)

            average_iterations.append(gnc.iterations)
            accuracy_percent.append(gnc.percentage - i)
        
        average_error_all.append(np.average(average_error))
        average_iterations_all.append(np.average(average_iterations))
        accuracy_percentage_all.append(np.abs(np.average(accuracy_percent)))

        if i % 10 == 0:
            print("Calculated up to", i, "percentage")

    
    fig, ax = plt.subplots(1)

    x_vals = np.arange(min_percentage, max_percentage+1, step)

    fig.suptitle("Num per percent: {}".format(num_per_percent), fontsize=9)

    ax.set_title("Average error")
    ax.set_xlim([min_percentage, max_percentage])
    ax.plot(x_vals, average_error_all)


    fig, ay = plt.subplots(1)
    ay.set_title("Average number of iterations")
    ay.set_xlim([min_percentage, max_percentage])
    ay.plot(x_vals, average_iterations_all)
    ay.set_ylim(ymin=0)
    
    # fig, az = plt.subplots(3)
    # az.set_title("Average difference on percentage outliers vs truth")
    # az.set_xlim([min_percentage, max_percentage])
    # az.plot(x_vals, accuracy_percentage_all)

    plt.show()



def showing_gnc_with_plot(num=50):
    print("Showing gnc with procrustes and plot")
    gnc = GNC_PROCRUSTES(num)
    gnc.calculate_perc()
    gnc.plot_o3d()
    print(gnc.n)

# ------------------------------------------


if __name__ == "__main__":
    
    # testing_gnc_procrustes(
    #     min_percentage =   1,
    #     max_percentage = 100,
    #     step=              1,
    #     num_per_percent=   5
    # )

    showing_gnc_with_plot(85)

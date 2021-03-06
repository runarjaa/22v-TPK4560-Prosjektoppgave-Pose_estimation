from cgitb import strong
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

import matplotlib.patheffects as pe

from utils import *

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})



class GNC_Average:
    def __init__(self) -> None:
        pass

    def load_points(self, cent, std, num):
        self.numbers = np.random.normal(cent, std, num)
        self.cent = cent
        self.std = std
        self.n = num

    def load_points_extra(self, cent, std, num, xmin, xmax, xnum):
        numbers = np.random.normal(cent, std, num-10)
        noise = np.random.randint(xmin, xmax, xnum)
        exact = np.full(10, cent)
        self.numbers = np.concatenate((numbers, noise, exact))
        self.cent = cent
        self.std = std
        self.n = self.numbers.shape[0]

    def gnc_average(self):
        # Initial guess
        x0 = 1.0

        # Loss function initialization
        self.r = np.zeros(self.n)
        for i in range(self.n):
            self.r[i] = np.linalg.norm(self.numbers[i] - x0)
        r0_max = np.max(self.r)

        # GNC initialization
        max_iterations = 1000
        eps = 0.1
        mu_update = 1.4
        self.w = np.ones(self.n)
        mu = eps**2 / (2*r0_max**2 - eps**2)

        self.last_iter = []
        self.x_iter = []
        self.iterations = 0

        # GNC iteration
        for i in range(max_iterations):
            # Information
            self.iterations += 1
            self.last_iter.append(np.sum(self.w)) 

            # Weigted average
            self.x = np.dot(self.numbers, self.w)/np.sum(self.w)
            self.x_iter.append(self.x)

            # Loss Function
            for j in range(self.n):
                self.r[j] = np.linalg.norm(self.numbers[j] - self.x)
                self.w[j] = w_from_r(self.r[j], eps, mu)
            
            mu = mu_update * mu

            # Stopping criteria
            if i >= 5:
                if np.sum(self.w) == self.last_iter[i]:
                    break
        
        self.inliers = []
        self.outliers = []

        for i, n in enumerate(self.numbers):
            if self.w[i] == 1.0:
                self.inliers.append(n)
            else:
                self.outliers.append(n)
        
        self.inlier_num = len(self.inliers)
        self.outlier_num = len(self.outliers)
        self.percentage = (1-self.inlier_num/(self.inlier_num+self.outlier_num))*100

    def print_results(self):
        print("Here are the results:")
        print("x =", self.x)
        print("Difference actual center", np.abs(self.x - self.cent))
        print()


    def plot_histogram(self):
        fig, plothist = plt.subplots()
        count, bins, ignored = plothist.hist(self.numbers, 70, density=True, label='Number Distribution')
        # plothist.plot(bins, 1/(self.std * np.sqrt(2 * np.pi)) *
        #        np.exp( - (bins - self.cent)**2 / (2 * self.std**2) ),
        #        linewidth=1, color='r')
        plothist.axvline(self.x, linewidth=3, color='orange', label='Solution')
        plothist.legend(loc="upper right")
        # plt.show()


if __name__ == "__main__":
    # Settings
    cent = 10.0
    std = 1.0
    num = 100
    outliers = 100
    min = cent + 10
    max = cent + 50

    # Information
    answer = []
    iterations = []
    percentage = []
    x_iter = []

    test_num = 300
    print("Testing GNC on weighted average of real numbers")
    print("Center:", cent, "\nNumber of numbers:", num, "\nIterations", test_num)
    for i in range(test_num):
        gnc = GNC_Average()
        # gnc.load_points(cent, std, num)
        gnc.load_points_extra(cent, std, num, min, max, outliers)
        gnc.gnc_average()

        answer.append(gnc.x)
        iterations.append(gnc.iterations)
        percentage.append(gnc.percentage)
        x_iter.append(gnc.x_iter)

        if i == 2:
            fig, ay = plt.subplots()
            _, test, __ = plt.hist(gnc.numbers, 100, density=True)
            # print(gnc.numbers)
            ay.hist(gnc.numbers, bins= 100)
            # plt.show()

    avrg_answer = np.sum(answer)/len(answer)
    avrg_iterat = np.sum(iterations)/len(iterations)
    avrg_percen = np.sum(percentage)/len(percentage)

    print("\nAverage iterations:\t", avrg_iterat)
    print("Average answer:\t\t", avrg_answer)
    print("Average percentage:\t", avrg_percen)

    gnc.plot_histogram()
    
    # Running gnc on answers for fun
    gnc = GNC_Average()
    gnc.numbers = np.array(answer)
    gnc.n = len(answer)
    gnc.gnc_average()

    print("\nTried doing gnc on the answers from the previous tests:")
    print(gnc.x)
    
    # Plotting
    fig, ax1 = plt.subplots(1)

    # ax1: Final answer
    ax1.set_title("Final answer")
    ax1.set_xlabel("Global Iteration")
    ax1.set_ylabel("X value")
    ax1.plot(answer, label='X value')
    ax1.axhline(np.sum(answer)/len(answer), color='orange', label = 'Average X value')
    ax1.axhline(cent, color='black', label = 'True center')
    ax1.legend(loc="upper right")
    # plt.show()

    fig, ax2 = plt.subplots(1)
    # ax2: Iteration of x 
    ax2.set_title("Iterations of x")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("X value")
    ax2.axhline(cent, color='black', linewidth = 1, label = 'True center')
    for i, n in enumerate(x_iter):
        ax2.plot(n, linewidth= 3)
    ax2.legend(loc="upper right")
    plt.show()
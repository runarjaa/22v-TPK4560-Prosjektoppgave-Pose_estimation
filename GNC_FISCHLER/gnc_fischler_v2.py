from utils import *

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': '{: 0.6f}'.format})



class GNC_LINEAR_REGRESSION():
    def __init__(self, n = 100, per = 50, fischler = False) -> None:
        self.n = n
        self.per = per
        self.fischler = fischler
        
        if fischler: self.load_fischler()
        else:        self.load_data()
        self.gnc_iteration()


    def load_data(self):
        self.min = 0
        self.max = 10
        self.a = 1
        self.b = 1

        self.inli = np.ceil(self.n - self.n * self.per/100).astype('int32')
        self.outl = np.floor(self.n * self.per/100).astype('int32')

        self.line_x  = np.linspace(self.min, self.max, self.inli).reshape((self.inli, 1))
        noise_x = np.random.uniform(self.min, self.max , self.outl).reshape((self.outl, 1))

        self.line_y  = np.asarray([self.y(x, self.a, self.b) for x in self.line_x]).reshape((self.inli, 1))
        noise_y = np.asarray([np.array(np.random.randint(0, 10)) for x in noise_x]).reshape((self.outl, 1))

        X = np.concatenate((self.line_x, noise_x), axis=0)
        self.Y = np.concatenate((self.line_y, noise_y), axis=0)

        self.X = np.concatenate((X, np.ones((self.n, 1))), axis=1)


    def load_fischler(self):
        points = np.array([
                    [0,0],
                    [1,1],
                    [2,2],
                    [3,2],
                    [3,3],
                    [4,4],
                    [10,2]
                ])
        self.n = points.shape[0]

        self.Y = np.zeros(self.n); self.X = np.ones((self.n,2))

        for i in range(self.n):
            self.Y[i] = points[i,1]; self.X[i] = np.array([points[i,0], 1])
        
        self.inli = 5
        self.outl = 2
        self.min = 0
        self.max = 10
        self.a = 1
        self.b = 0


    def gnc_iteration(self):
        # Initiation
        m0 = 0.
        b0 = 0.
        self.B0 = np.array([m0, b0])

        r = np.zeros(self.n)
        for i in range(self.n):
            r[i] = np.linalg.norm(self.Y[i] - self.X[i] @ self.B0)
        r0_max = np.max(r)

        max_iter = 1000
        eps = 0.011
        mu_update = 1.4
        w = np.ones(self.n, dtype=np.float32)
        mu = eps**2 / (2*r0_max**2 - eps**2)

        # Iteration
        self.last_iter = []
        self.B_iter = []
        self.iterations = 0

        for i in range(max_iter):
            self.iterations += 1
            self.last_iter.append(np.sum(w))

            W = np.diag(w)
            self.B = np.linalg.inv(self.X.T @ W @ self.X) @ self.X.T @ W @ self.Y

            self.B_iter.append(self.B)

            for j in range(self.n):
                r[j] = np.linalg.norm(self.Y[j] - self.X[j] @ self.B)
                w[j] = w_from_r(r[j], eps, mu)

            mu = mu_update * mu

            if i >= 5:
                if np.sum(w) == self.last_iter[i]:
                    self.last_iter.append(np.sum(w))
                    break    

    def plot_results(self):
        fig, ax = plt.subplots()
        # ax.set_xlim([-0.5,10.5]);ax.set_ylim([-0.4,5])
        x = np.linspace(self.min-5, self.max+5, self.n*10)
        ax.plot(x, self.B[0]*x + self.B[1], color='blue', linewidth=3.5, label='Guessed line')
        ax.plot(x, self.a*x + self.b, color='orange', label='Actual Line')
        ax.legend(loc='upper left')
        ax.scatter(self.X[:,0], self.Y, c='g')

        if self.fischler:
            ax.set_xlim([-0.5,10.5]);ax.set_ylim([-0.4,5])

        plt.show()

    def y(self, x, m, b):
        return m*x + b
# ------------------------------------------


# ----------------- Testing ----------------
def testing_gnc_LinReg(
            num_of_num      = 100,
            min_percentage  =   1,
            max_percentage  = 100,
            step            =   1,
            num_per_percent =   1
        ):
    
    print("Testing Linear Regression with gnc")

    average_error_all = []
    average_iterations_all = []
    accuracy_percentage_all = []

    for i in range(min_percentage, max_percentage+1, step):
        average_error = []
        average_iterations = []
        accuracy_percent = []
        for j in range (num_per_percent):
            
            gnc = GNC_LINEAR_REGRESSION(n=num_of_num, per=i)

            error = np.linalg.norm(gnc.B0 - gnc.B)
            average_error.append(error)

            average_iterations.append(gnc.iterations)
            accuracy_percent.append(gnc.last_iter[-1] - i)

            if i == 60:
                gnc.plot_results()

            if i == min_percentage:
                print("X shape:", gnc.X.shape)
                print("Y shape:", gnc.Y.shape)
                print("B shape:", gnc.B.shape)
        
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

    # fig, ay = plt.subplots(1)
    # ay.set_title("Average number of iterations")
    # ay.set_xlim([min_percentage, max_percentage])
    # ay.plot(x_vals, average_iterations_all)
    # ay.set_ylim([0, 100])
    
    # fig, az = plt.subplots(1)
    # az.set_title("Average difference on percentage outliers vs truth")
    # az.set_xlim([min_percentage, max_percentage])
    # az.plot(x_vals, accuracy_percentage_all)
    

    plt.show()


def testing_gnc_LinReg_Fishler():
    gnc = GNC_LINEAR_REGRESSION(fishler=True)
    gnc.plot_results()
# ------------------------------------------

if __name__ == "__main__":

    testing_gnc_LinReg(
        num_of_num     = 100,
        min_percentage =   1,
        max_percentage = 100,
        step           =   5,
        num_per_percent=   1
    )

    # testing_gnc_LinReg_Fishler()
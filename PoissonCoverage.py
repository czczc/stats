# calculate coverage for Poisson distributed data
# see https://www-cdf.fnal.gov/physics/statistics/notes/cdf6438_coverage.pdf
import numpy as np

class PoissonCoverage:
    def __init__(self, name, delta=1, N=30):
        self.name = name
        self.delta = delta
        self.N = N
        self.measurement = np.arange(self.N, dtype=np.float32)
        self.interval_low = np.zeros(self.N, dtype=np.float32)
        self.interval_high = np.zeros(self.N, dtype=np.float32)

        self.CalculateIntervals()
        self.CalculateCoverage()
        self.Plot()
    
    def CalculateIntervals(self):
        if self.name == 'Pearson':
            self.Pearson()
        elif self.name == 'Neyman':
            self.Neyman()
        elif self.name == 'CNP':
            self.CNP()
        elif self.name == 'Likelihood':
            self.Likelihood()

    def Pearson(self):
        d = self.delta
        n = self.measurement
        s = np.sqrt(n*d+d*d/4)
        self.interval_low = n + d/2 - s
        self.interval_high = n + d/2 + s

    def Neyman(self):
        d = self.delta
        n = self.measurement
        s = np.sqrt(n*d)
        self.interval_low = n - s
        self.interval_high = n + s
    
    def CNP(self):
        d = self.delta
        for i in range(self.N):
            n = self.measurement[i]
            coeff = [1, 0, -3*(n*n+d*n), 2*n*n*n]
            roots = np.sort(np.roots(coeff))
            self.interval_low[i] = roots[1]
            self.interval_high[i] = roots[2]
            
    def Likelihood(self):
        from scipy.optimize import fsolve
        d = self.delta
        for i in range(self.N):
            n = self.measurement[i]
            def f(x):
                if x<=0 or n<=0:
                    return 0.1
                return x - n*np.log(x) + n*np.log(n) - n - d/2
            if n == 0:
                self.interval_low[i] = 0
                self.interval_high[i] = fsolve(f, n+0.5)
            else:
                self.interval_low[i] = fsolve(f, n-0.5)
                self.interval_high[i] = fsolve(f, n+0.5)

    def CalculateCoverage(self):
        self.mu = np.arange(0, self.N-10, 0.1)
        self.prob = np.zeros(len(self.mu))
        for j in range(len(self.mu)):
            mu = self.mu[j]
            for i in range(self.N):
                n = self.measurement[i]
                if mu > self.interval_low[i] and mu <= self.interval_high[i]:
                    self.prob[j] += np.power(mu, n) * np.exp(-mu) / np.math.factorial(n)
                    # print(mu, self.interval_low[i], self.interval_high[i], self.prob[j])
    
    def Plot(self):
        import matplotlib
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.scatter(self.mu[1:], self.prob[1:], s=2**2)
        ax.set(xlabel=r'$\mu$', ylabel='coverage', title=self.name)
        ax.set_ylim([0,1])
        ax.set_xlim([0,20])
        plt.axhline(y=0.683, color='r', linestyle='-', linewidth=1)

        # ax.grid()
        fig.savefig(self.name+'.pdf')
        plt.show()


    def Print(self):
        print(self.name)
        for i in range(self.N):
            print('{:6.2f} {:6.2f} {:6.2f}'.format(
                self.measurement[i], 
                self.interval_low[i], 
                self.interval_high[i])
            )
        # print(self.mu)
        # print(self.prob)


if __name__ == "__main__":
    for name in ['Pearson', 'Neyman', 'CNP', 'Likelihood']:
    # for name in ['Pearson']:
        x = PoissonCoverage(name)
        x.Print()
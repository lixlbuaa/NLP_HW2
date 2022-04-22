import numpy as np
import matplotlib.pyplot as plt

class Coin():
    def __init__(self, n_epoch=100):
        self.n_epoch = n_epoch
        self.params = {'s1': None, 's2': None, 'pi': None, 'p':None, 'q':None, 'mu': None, 'mu2': None}

    def __init_params(self, n):
        self.params = {
                       's1': [0.2],
                       's2': [0.4],
                       'pi': [0.8],
                       'p': [0.7],
                       'q': [0.6],
                       'mu2': np.random.rand(n),
                       'mu': np.random.rand(n)}

        self.ele = {
            's1':[0 for x in range(0, self.n_epoch+1)],
            's2': [0 for x in range(0, self.n_epoch+1)],
            'pi': [0 for x in range(0, self.n_epoch+1)],
            'p': [0 for x in range(0, self.n_epoch+1)],
            'q': [0 for x in range(0, self.n_epoch+1)],
        }

    def E_step(self, y, n):
        pi = self.params['pi'][0]
        p = self.params['p'][0]
        q = self.params['q'][0]
        s1 = self.params['s1'][0]
        s2 = self.params['s2'][0]

        for i in range(n):
            self.params['mu'][i] = (s1 * pow(pi, y[i]) * pow(1-pi, 1-y[i])) / \
                                   (s1 * pow(pi, y[i]) * pow(1-pi, 1-y[i]) + s2 * pow(p, y[i]) * pow(1-p, 1-y[i])
                                    + (1-s1-s2) * pow(q,y[i])*pow(1-q,1-y[i]))
            self.params['mu2'][i] = (s2 * pow(p, y[i]) * pow(1-p, 1-y[i])) /\
                                    (s1 * pow(pi, y[i]) * pow(1-pi, 1-y[i]) + s2 * pow(p, y[i]) * pow(1-p, 1-y[i])
                                     + (1-s1-s2) * pow(q,y[i])*pow(1-q,1-y[i]))

    def M_step(self, y, n):
        mu = self.params['mu']
        mu2 = self.params['mu2']

        self.params['s1'][0] = sum(mu) / n
        self.params['s2'][0] = sum(mu2) / n
        self.params['pi'][0] = sum([mu[i] * y[i] for i in range(n)]) / sum(mu)
        self.params['p'][0] = sum([mu2[i] * y[i] for i in range(n)]) / sum(mu2)
        self.params['q'][0] = sum([(1-mu[i]-mu2[i]) * y[i] for i in range(n)]) / \
                              sum([1-mu_i-mu2_i for mu_i,mu2_i in zip(mu,mu2)])

    def fit(self, y):
        n = len(y)
        self.__init_params(n)
        print(0, self.params['s1'], self.params['s2'], self.params['pi'], self.params['p'], self.params['q'])
        flag_begin = self.params['s1'][0] * self.params['pi'][0] + self.params['s2'][0] * self.params['p'][0] + (
                    1 - self.params['s1'][0] - self.params['s2'][0]) * self.params['q'][0]

        self.ele['s1'][0] = self.params['s1'][0]
        self.ele['s2'][0] = self.params['s2'][0]
        self.ele['pi'][0] = self.params['pi'][0]
        self.ele['p'][0] = self.params['p'][0]
        self.ele['q'][0] = self.params['q'][0]

        print(f'begin ---{flag_begin}')
        for i in range(self.n_epoch):
            self.E_step(y, n)
            self.M_step(y, n)
            print(i+1, self.params['s1'], self.params['s2'], self.params['pi'], self.params['p'], self.params['q'])
            self.ele['s1'][i+1] = self.params['s1'][0]
            self.ele['s2'][i+1] = self.params['s2'][0]
            self.ele['pi'][i+1] = self.params['pi'][0]
            self.ele['p'][i+1] = self.params['p'][0]
            self.ele['q'][i+1] = self.params['q'][0]

model = Coin()
dataset = np.random.randint(0, 2, 1000)
model.fit(dataset)


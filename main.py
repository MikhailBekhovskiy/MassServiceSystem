from matplotlib import pyplot as plt

def summarize(vector):
    sum = 0.
    for component in vector:
        sum += component
    return sum

def normalize(vector):
    n = len(vector)
    new_vector = [0] * n
    sum = summarize(vector)
    for i in range(n):
        new_vector[i] = vector[i] / sum
    return new_vector

def probabilize(vector):
    n = len(vector)
    sma = min(vector)
    new_vector = [0] * n
    for i in range(n):
        new_vector[i] = vector[i] - sma
    return normalize(new_vector)


class MassServiceSystem:
    def __init__(self, inflow=10, outflow=1, servers=10, qSize=5):
        self.lamb = inflow
        self.mu = outflow
        self.servs = servers
        self.queue = qSize
        self.states = servers + qSize + 1
        self.iv = [1] + [0] * (self.states - 1)
    
    def RHS(self, p_t: list) -> list:
        diff = [0] * len(p_t)
        diff[0] = -self.lamb * p_t[0] + self.mu * p_t[1]
        for i in range(1, self.servs + 1):
            diff[i] = self.lamb * p_t[i-1] + (i + 1) * self.mu*p_t[i+1] - (self.lamb + i*self.mu) * p_t[i]
        for i in range(self.servs + 1, self.states - 1):
            diff[i] = self.lamb * p_t[i-1] + self.servs * self.mu*p_t[i+1] - (self.lamb + self.servs*self.mu) * p_t[i]
        diff[self.states - 1]=-self.servs*self.mu*p_t[self.states - 1] + self.lamb * p_t[self.states - 2]
        return diff

    def ode45(self, time_step=0.01, time=100):
        sol = [self.iv]
        times = [0.]
        steps = int(time / time_step)
        tmp = [0.] * self.states
        #print(sol)
        for t in range(1, steps):
            k1 = self.RHS(p_t=sol[t-1])
            #print(k1)
            k2 = list()
            k3 = list()
            k4 = list()
            for i in range(self.states):
                k2.append(sol[t-1][i] + 0.5 * time_step* k1[i])
            k2 = self.RHS(p_t=k2)
            for i in range(self.states):
                k3.append(sol[t-1][i] + 0.5 * time_step* k2[i])
            k3 = self.RHS(p_t=k3)
            for i in range(self.states):
                k4.append(sol[t-1][i] + time_step * k3[i])
            k4 = self.RHS(p_t=k4)
            for i in range(self.states):
                tmp[i] = sol[t-1][i] + (time_step/6) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
            sol.append(probabilize(tmp))
            times.append(times[t-1] + time_step)
        return sol, times

    def plot_state(self, state=0, time=50):
        sol, x = self.ode45(time=time)
        y = list()
        for i in range(len(sol)):
            y.append(sol[i][state])
        plt.plot(x, y)

    def save_state(self, state=0, time=50):
        self.plot_state(state=state, time=time)
        plt.savefig(f'plot_lambda{self.lamb}_mu{self.mu}_p{state}.png')
        plt.clf()

    def plot_all(self, time=50):
        for state in range(self.states):
            self.plot_state(state=state, time=time)
        plt.savefig(f'plot_all_lambda{self.lamb}_mu{self.mu}.png')
        plt.clf()

    def states_to_requests(self, states):
        requests = states.copy()
        for i in range(len(requests)):
            requests[i] = min((self.servs, i))
        return requests





if __name__=='__main__':
    smo = MassServiceSystem()
    time = 50
    smo.plot_all(time=time)
    #sol = smo.ode45(time=time_steps)
    #print(sol)
    '''for i in range(smo.states):    
        smo.plot_state(state=i, time=time_steps)
        plt.clf()'''
    #plt.plot(requests)
    #plt.show()
    # print(len(smo.iv))
    '''distr = ode45(time=1000)
    states = list()
    for i in range(len(distr)):
        states.append(distr[i].index(max(distr[i])))
    print(states)'''
    #print(distr)
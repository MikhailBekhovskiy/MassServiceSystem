from matplotlib import pyplot as plt
from numpy import random

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
    def __init__(self, inflow=10 , outflow=5, servers=10, qSize=5):
        self.lamb = inflow
        self.mu = outflow
        self.servs = servers
        self.queue = qSize
        self.states = servers + qSize + 1
        self.iv = [1.] + [0.] * (self.states - 1)
    
    def RHS(self, p_t: list) -> list:
        diff = [0.] * len(p_t)
        diff[0] = -self.lamb * p_t[0] + self.mu * p_t[1]
        for i in range(1, self.servs + 1):
            diff[i] = self.lamb * p_t[i-1] + (i + 1) * self.mu*p_t[i+1] - (self.lamb + i*self.mu) * p_t[i]
        for i in range(self.servs + 1, self.states - 1):
            diff[i] = self.lamb * p_t[i-1] + self.servs * self.mu*p_t[i+1] - (self.lamb + self.servs*self.mu) * p_t[i]
        diff[self.states - 1]=-self.servs*self.mu*p_t[self.states - 1] + self.lamb * p_t[self.states - 2]
        return diff

    def ode45(self, time_step=0.01, time=5):
        sol = [self.iv]
        times = [0.]
        steps = int(time / time_step)
        #print(sol)
        for t in range(1, steps):
            k1 = self.RHS(p_t=sol[t-1])
            #print(k1)
            k2 = [0.] * self.states
            k3 = [0.] * self.states
            k4 = [0.] * self.states
            for i in range(self.states):
                k2[i] = sol[t-1][i] + 0.5 * time_step* k1[i]
            k2 = self.RHS(p_t=k2)
            for i in range(self.states):
                k3[i] = sol[t-1][i] + 0.5 * time_step* k2[i]
            k3 = self.RHS(p_t=k3)
            for i in range(self.states):
                k4[i] = sol[t-1][i] + time_step * k3[i]
            k4 = self.RHS(p_t=k4)
            sol.append(list())
            for i in range(self.states):
                sol[t].append(sol[t-1][i] + (time_step/6) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]))
            #sol[t] = probabilize(sol[t])
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
        plt.savefig(f'plot_lambda{self.lamb}_mu{self.mu}_p{state}_t{time}_s{self.servs}_q{self.queue}.png')
        plt.clf()

    def plot_all(self, time=50):
        for state in range(self.states):
            self.plot_state(state=state, time=time)
        plt.savefig(f'plot_all_lambda{self.lamb}_mu{self.mu}_t{time}_s{self.servs}_q{self.queue}.png')
        plt.clf()

    def states_to_requests(self, states):
        requests = states.copy()
        for i in range(len(requests)):
            requests[i] = min((self.servs, i))
        return requests

    # imitation modelling utility functions
    def gen_inflow(self, time=5, step=0.01):
        steps = int(time/step)
        reqs = [0] * steps
        for i in range(1, steps):
            if random.exponential(scale=1/self.lamb) >= 0.5:
                reqs[i] = reqs[i-1] + 1
            else:
                reqs[i] = reqs[i-1]
        return reqs    

    # one imitation run
    # 1. generate array modelling inflow of requests (sample exponential distr scaled by inverse inflow intensity),
    #    since new arrivals are not dependant on service
    # 2. at each timestep:
    #   2.1 imitate service event (sample exponential distr scaled by inverse outflow intensity multiplied by servers at work)
    #   2.2 change states if needed (if the queue is filled, new requests are not accounted for i.e. discarded):
    #       2.2.1a 0-state, new arrival -> 1-state
    #       2.2.1b 0-state, no new arrivals -> 0-state
    #       2.2.2 !0-state, new arrivals, no service, queue not filled -> next state
    #       2.2.3 !0-state, no new arrival, service -> prev state
    #       2.2.4 !0-state, (new arrival + service) or (no arrival no service) -> cur state
    # N.B.: sampling input event at each timestep instead of preceding generation is useless, since we don't employ parallelism
    # NumPy friends probably would benefit, though)
    def imitation_run(self, time=5, step=0.01):
        reqs = self.gen_inflow(time, step)
        sol = [[1] + [0] * (self.states - 1)]
        times = [0.] * len(reqs)
        state = 0
        for t in range(1, len(reqs)):
            sol.append([0] * self.states)
            if state == 0:
                if reqs[t] > reqs[t-1]:
                    sol[t][1] = 1
                    state = 1
                else:
                    sol[t][0] = 1
            else: 
                if state < self.servs:
                    serve_scale = 1/state/self.mu
                else:
                    serve_scale = 1/self.servs/self.mu
                isServed = random.exponential(scale=serve_scale) >= 0.5
                if reqs[t] == reqs[t-1] and isServed:
                    state -= 1
                    sol[t][state] = 1
                elif reqs[t] > reqs[t-1] and not isServed and state < (self.states - 1):
                    state += 1
                    sol[t][state] = 1
                else:
                    sol[t][state] = 1
            times[t] = times[t-1] + step
        return sol, times

if __name__=='__main__':
    smo = MassServiceSystem()
    time = 100
    reqA = smo.gen_inflow(time=time)
    plt.plot(reqA)
    plt.show()
    #smo.plot_all(time=time)
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
from matplotlib import pyplot as plt
from numpy import random


# Vector-Matrix Utilities
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

def add(matrA, matrB):
    n = len(matrA)
    m = len(matrA[0])
    if len(matrB) != n or len(matrB[0]) != m:
        print("Faulty matrices")
        raise ValueError
    matrC = list()
    for i in range(len(matrA)):
        matrC.append(list())
        for j in range(len(matrA[0])):
            matrC[i].append(matrA[i][j] + matrB[i][j])
    return matrC


def scale(matrA, scale):
    matrC = list()
    for i in range(len(matrA)):
        matrC.append(list())
        for j in range(len(matrA[0])):
            matrC[i].append(scale * matrA[i][j])
    return matrC


# MassServiceSystem model and specific methods
class MassServiceSystem:
    def __init__(self, inflow=10 , outflow=5, servers=10, qSize=5):
        self.lamb = inflow
        self.mu = outflow
        self.serv_num = servers
        self.queue = qSize
        self.states = servers + qSize + 1
        self.iv = [1.] + [0.] * (self.states - 1)
        # Imitation specific attributes
        self.servers = dict()
        for i in range(self.serv_num):
            self.servers[i+1] = {
                'busy': False,
                'until': 0.
            }
        self.reqs_num = 0
        self.unassigned = 0
    #--------------------------------------------------
    #--------------------------------------------------
    # TRADITIONAL
    # model-based calculation utility - diff. eqs right-handed sides
    def RHS(self, p_t: list) -> list:
        diff = [0.] * len(p_t)
        diff[0] = -self.lamb * p_t[0] + self.mu * p_t[1]
        for i in range(1, self.serv_num + 1):
            diff[i] = self.lamb * p_t[i-1] + (i + 1) * self.mu*p_t[i+1] - (self.lamb + i*self.mu) * p_t[i]
        for i in range(self.serv_num + 1, self.states - 1):
            diff[i] = self.lamb * p_t[i-1] + self.serv_num * self.mu*p_t[i+1] - (self.lamb + self.serv_num*self.mu) * p_t[i]
        diff[self.states - 1]=-self.serv_num*self.mu*p_t[self.states - 1] + self.lamb * p_t[self.states - 2]
        return diff

    # model-based calculation, the most common Runge-Kutta method
    def ode45(self, step=0.01, time=5.):
        sol = [self.iv]
        times = [0.]
        steps = int(time / step)
        #print(sol)
        for t in range(1, steps + 1):
            k1 = self.RHS(p_t=sol[t-1])
            #print(k1)
            k2 = [0.] * self.states
            k3 = [0.] * self.states
            k4 = [0.] * self.states
            for i in range(self.states):
                k2[i] = sol[t-1][i] + 0.5 * step* k1[i]
            k2 = self.RHS(p_t=k2)
            for i in range(self.states):
                k3[i] = sol[t-1][i] + 0.5 * step* k2[i]
            k3 = self.RHS(p_t=k3)
            for i in range(self.states):
                k4[i] = sol[t-1][i] + step * k3[i]
            k4 = self.RHS(p_t=k4)
            sol.append(list())
            for i in range(self.states):
                sol[t].append(sol[t-1][i] + (step/6) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]))
            #sol[t] = probabilize(sol[t])
            times.append(times[t-1] + step)
        return sol, times

    #--------------------------------------------
    #--------------------------------------------
    # IMITATION
    # imitation modelling utility functions

    # flushing previous run data, for total experiment
    def clear_system(self):
        self.reqs_num = 0
        self.unassigned = 0
        for i in range(1, self.serv_num + 1):
            self.servers[i]['busy'] = False
            self.servers[i]['until'] = 0.

    # lookup freeserver for assignment
    # either when new request arrives, or when servers get free
    # and there are unassigned tasks
    def get_free_server(self):
        for s in range(1, self.serv_num + 1):
            if not self.servers[s]['busy']:
                return s
        return -1

    # assigning task to a server with calculation of 
    # serving time        
    def assign_task(self, serv, cur_time):
        self.servers[serv]['busy'] = True
        self.servers[serv]['until'] = cur_time + random.exponential(1/self.mu)

    # free server when it completes requests or while flushing
    def free_server(self, serv):
        self.servers[serv]['busy'] = False
        self.servers[serv]['until'] = 0.
        self.reqs_num -= 1

    # when new request arrives
    def receive_request(self, cur_time):
        if self.reqs_num < (self.serv_num + self.queue):
            self.reqs_num += 1
            if self.reqs_num <= self.serv_num:
                self.assign_task(serv=self.get_free_server(), cur_time=cur_time)
            else:
                self.unassigned += 1
    
    # update server state (free and assign unassigned tasks)
    # when we move forward in time
    def check_on_servs(self, cur_time):
        for i in range(1, self.serv_num + 1):
            if self.servers[i]['busy']:
                if self.servers[i]['until'] <= cur_time:
                    self.free_server(i)
        while self.unassigned > 0 and (self.reqs_num - self.unassigned) < self.serv_num:
            self.assign_task(self.get_free_server(), cur_time=cur_time)
            self.unassigned -= 1
        
    # one imitation run
    # returns translated runtime data to gridpoints 
    # for comparison with Kolmogorov eqs' integration
    def imitation_run(self, time=5., step=0.1):
        steps = int(time/step)
        i_sol = [[0.] * self.states]
        for k in range(steps):
            i_sol.append([0] * self.states)
        i_sol[0][0] = 1
        next_write = step
        grid_point = 0
        cur_time = 0.
        while cur_time < time:
            while cur_time >= next_write:
                i_sol[grid_point + 1][self.reqs_num] = 1.
                grid_point += 1
                next_write += step
            next_req = random.exponential(1/self.lamb)
            cur_time += next_req
            self.check_on_servs(cur_time)
            self.receive_request(cur_time)
            #print(self.reqs_num)
        while grid_point < steps:
            i_sol[grid_point + 1][self.reqs_num] = 1.
            grid_point += 1
        return i_sol

    # multiple runs of imitation modelling, averaged
    def imitation_experiment(self, time=5., step=0.01, runs=1000):
        Sol = self.imitation_run(time=time, step=step)
        # print(Sol)
        for r in range(1, runs):
            self.clear_system()
            sol = self.imitation_run(time=time, step=step)
            #print(Sol[0])
            Sol = add(Sol, sol)
            # print(Sol[0])
        # print(Sol)
        Sol = scale(Sol, float(1/runs))
        # print(Sol[0])
        return Sol
    #------------------------------------------
    #------------------------------------------
    # PLOTTING
    def plot_state(self, sol, times, state=0):
        x = times
        y = list()
        for i in range(len(sol)):
            y.append(sol[i][state])
        plt.plot(x, y, label=f'p{state}')

    def plot_all(self, sol, time, times, isImit, isFull=True, imiRuns=-1, step=-1):
        for state in range(self.states):
            self.plot_state(sol=sol, times=times, state=state)
        plt.legend()
        if not isFull:
            plt.savefig(f'imit_1run_lambda{self.lamb}_mu{self.mu}_s{self.serv_num}_q{self.queue}_t{time}.png')
        elif isImit and isFull:
            plt.savefig(f'imit_{imiRuns}runs_lambda{self.lamb}_mu{self.mu}_s{self.serv_num}_q{self.queue}_t{time}.png')
        else:
            plt.savefig(f'trad_lambda{self.lamb}_mu{self.mu}_s{self.serv_num}_q{self.queue}_t{time}_h{step}.png')
        plt.clf()

if __name__=='__main__':
    # initiating model with parameters
    smo = MassServiceSystem(inflow=5, outflow=1, servers=10, qSize=5)
    time = 5
    step = 0.1
    imiRuns = 10000
    # running
    t_sol, times = smo.ode45(step=step,time=time)
    i_sol = smo.imitation_experiment(time=time, step=step, runs=imiRuns)
    #print(i_sol)
    #petit_sol = smo.imitation_run(time=time, step=step)
    #print(petit_sol[0])
    smo.plot_all(t_sol, time=time, times=times, isImit=False, step=step)
    smo.plot_all(i_sol, time=time, times=times, isImit=True, imiRuns=imiRuns, step=step)
    #smo.plot_all(petit_sol, times, True, False)

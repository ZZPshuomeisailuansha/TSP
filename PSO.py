import numpy as np
from graph import Graph
import matplotlib.pyplot as plt
class ParticleGroup:
    def __init__(self,graph:Graph,num=50,Pbestr=0.7,Gbestr=0.1):
        self.num = num
        self.graph = graph
        self.group = []
        self.Gbest_fitness = 1e9
        for i in range(num):
            p = Particle(num=graph.num,graph=graph,Pbestr=Pbestr,Gbestr=Gbestr)
            self.group.append(p)
            if self.Gbest_fitness > p.Pbest_fitness:
                self.Gbest_fitness = p.Pbest_fitness
                self.Gbest = p.path
        

    def update(self):
        for i in range(self.num):
            self.group[i].update(self.Gbest)
            if self.Gbest_fitness > self.group[i].Pbest_fitness:
                self.Gbest_fitness = self.group[i].Pbest_fitness
                self.Gbest = self.group[i].path
class Particle:
    def __init__(self,num:int,graph:Graph,Pbestr=0.7,Gbestr=0.1):
        self.num = num
        self.graph = graph
        self.path = np.arange(num)
        np.random.shuffle(self.path)
        self.Pbest = self.path
        self.Pbest_fitness = self.fitness()
        self.Pbestr=Pbestr
        self.Gbestr=Gbestr
    # 粒子的适应度函数
    def fitness(self):
        result = 0
        for i in range(self.num-1):
            result += self.graph.distance_matrix[self.path[i]][self.path[i+1]]
        result += self.graph.distance_matrix[self.path[-1]][self.path[0]]
        return result
    # 计算交换序列
    def get_ss(self,p):
        ss = []
        x = self.path.copy()
        for i in range(len(x)):
            if x[i] != p[i]:
                j = np.where(x == p[i])[0][0]
                ss.append((i,j))
                x[i],x[j] = x[j],x[i]
        return ss
    # 进行交换操作,r为概率
    def Swap(self,ss,r):
        for i,j in ss:
            rand = np.random.random()
            if rand <= r:
                self.path[i],self.path[j] = self.path[j],self.path[i]

    def update(self,Gbest):
        #位置更新
        Gss = self.get_ss(Gbest)
        Pss = self.get_ss(self.Pbest)
        self.Swap(Gss,self.Gbestr)
        self.Swap(Pss,self.Pbestr)
    
        fitness = self.fitness()
        if fitness < self.Pbest_fitness:
            self.Pbest_fitness = fitness
            self.Pbest = self.path
        
class PSO:
    def __init__(self,city_num=10,particle_num=50,Gbestr=0.6,Pbestr=0.7,randomstate = 0,graph:Graph=None):
        if graph == None:
            self.graph= Graph(city_num,randomstate=randomstate)
        else:
            self.graph = graph
        self.city_num = city_num
        self.particleGroup = ParticleGroup(graph=self.graph,num=particle_num,Gbestr=Gbestr,Pbestr=Pbestr)
        self.path = self.particleGroup.Gbest
        self.path_fitness = self.particleGroup.Gbest_fitness
        self.result=[]
        self.result.append(self.path_fitness)
    def update(self,epoch):
        for i in range(epoch):
            self.particleGroup.update()
            self.path = self.particleGroup.Gbest
            self.path_fitness = self.particleGroup.Gbest_fitness
            self.result.append(self.path_fitness)

    def plot(self):
        self.graph.plotPath(self.path)
        epoch = np.arange(len(self.result))
        plt.figure()
        plt.plot(epoch,self.result)
        plt.show()


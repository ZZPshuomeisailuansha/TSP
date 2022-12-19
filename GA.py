import numpy as np
from graph import Graph
import matplotlib.pyplot as plt

class GA:
    def __init__(self,city_num=20,group_num=50,group_n=3,cross_r=0.7,mutation_r=0.3,randomstate=0,graph:Graph=None):
        '''
        city_num 城市数目
        group_num 种群数目
        group_n 进行n元选
        cross_r 交叉概率
        mutation_r 变异概率
        randomstate 随机种子
        graph 是否使用指定图像,为None则随机生成
        '''
        if graph == None:
            self.graph = Graph(city_num,randomstate=randomstate)
        else:
            self.graph = graph
        self.city_num = city_num
        self.group_num = group_num
        #生成种群
        self.group = Group(graph=self.graph,num=group_num,n=group_n,cross_r=cross_r,mutation_r=mutation_r)
        self.path = self.group.path                   #种群最优路径  
        self.distance = self.group.result             #最优路径的总长度
        self.result = []                              #记录遗传过程的结果变化
        self.result.append(self.group.result)

    #遗传，epoch为代数
    def update(self,epoch):
        for i in range(epoch):
            self.group.update()
            self.result.append(self.group.result)
        self.path = self.group.path
        self.distance = self.group.result
    
    def plot(self):
        self.graph.plotPath(self.path)
        epoch = np.arange(len(self.result))
        plt.figure()
        plt.plot(epoch,self.result)
        plt.show()


class Group:
    def __init__(self,graph:Graph,num,n=3,cross_r=0.7,mutation_r=0.1):
        '''
        graph 种群对应的图
        num 种群数目
        n 选择时使用n元选
        '''
        self.graph = graph
        self.num = num
        self.n = n
        self.cross_r = cross_r
        self.mutation_r = mutation_r
        self.group = []
        self.result = 1e9
        # 生成种群
        for i in range(num):
            g = Gine(graph=graph,num=graph.num,cross_r=cross_r,mutation_r=mutation_r)
            self.group.append(g)
            if self.result > g.fitness_:
                self.result = g.fitness_
                self.path = g.path
    #交叉操作,x和y是Gine类
    def crossover(self,x,y):
        rand = np.random.random()
        if rand<self.cross_r:
            #生成交叉的两个位点
            key_1 = np.random.randint(0,self.graph.num)
            key_2 = np.random.randint(0,self.graph.num)
            while key_1 == key_2:
                key_2 = np.random.randint(1,self.graph.num)
            key_1,key_2 = min(key_1,key_2),max(key_1,key_2)

            #截取交叉的片段
            r1 = y.path[key_1:key_2]
            r2 = x.path[key_1:key_2]
            
            # 删去与交叉片段相同的部分
            x.path = np.delete(x.path,np.where(np.isin(x.path,r1)))
            y.path = np.delete(y.path,np.where(np.isin(y.path,r2)))
            
            # 把交叉片段加入开头
            x.path = np.append(r1,x.path)
            y.path = np.append(r2,y.path)
            
        return x,y

    # 选择策略，这里使用的是锦标赛n元选
    def choose(self):
        group = []
        for i in range(self.num):
            # 随机从现有种群中选择n个个体
            Temp = np.random.randint(0,self.num,size=(self.n,))
            temp = 1e9
            # 选出距离最短的个体
            for j in Temp:
                self.group[j].fitness()
                if self.group[j].fitness_ < temp:
                    key =self.group[j].copy()
            group.append(key)
        self.group = group

    # 种群繁衍
    def update(self):
        # 依次进行选择，交叉，变异
        self.choose()

        for i in range(self.num):
            x = np.random.randint(0,self.num)
            y = np.random.randint(0,self.num)
            while x == y:
                y = np.random.randint(0,self.num)
            self.group[x],self.group[y] = self.crossover(self.group[x],self.group[y])
        
        for i in self.group:
            i.mutation()
        
        #更新每个个体的适应度，并更新种群的最优解
        for g in self.group:
            g.fitness()
            if self.result > g.fitness_:
                self.result = g.fitness_
                self.path = g.path
        
        #这步可能会加速收敛
        #self.group[0].path = self.path
        #self.group[0].fitness()
    
    # 打印出种群
    def Print(self):
        for i in self.group:
            print(i.path,i.fitness_)
#个体类
class Gine:
    def __init__(self,graph:Graph,num,cross_r=0.7,mutation_r=0.1,copy=False) :
        '''
        graph 对应的图
        num 个体的维数(其实就是graph的num)
        copy 是否为复制已存在的个体
        '''
        self.graph = graph
        self.num = num
        self.cross_r = cross_r
        self.mutation_r = mutation_r

        if copy == False:
            self.path = np.arange(num)
            np.random.shuffle(self.path)
            self.fitness_ = self.fitness()

    # 复制操作
    def copy(self):
        new = Gine(self.graph,self.num,self.cross_r,self.mutation_r,copy=True)
        new.path = self.path.copy()
        new.fitness_ = self.fitness_
        return new

    # 变异操作
    def mutation(self):
        rand = np.random.random()
        if rand<self.mutation_r:
            # 生成两个交换位点
            i = np.random.randint(0,self.num)
            j = np.random.randint(0,self.num)
            while i==j:
                j = np.random.randint(0,self.num)
            self.path[i],self.path[j] = self.path[j],self.path[i]
    # 计算总距离
    def fitness(self):
        result = 0
        for i in range(self.num-1):
            result += self.graph.distance_matrix[self.path[i]][self.path[i+1]]
        result += self.graph.distance_matrix[self.path[-1]][self.path[0]]
        self.fitness_ = result
        return result

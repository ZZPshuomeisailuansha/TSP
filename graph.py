import numpy as np
import matplotlib.pyplot as plt
class Graph:
    def __init__(self,num=10,randomstate=0):
        self.num = num
        np.random.seed(randomstate)
        # 生成数量为num的点
        self.location = np.random.rand(num,2)
        self.distance_matrix = self.distanceMatrix()
    # 计算距离矩阵
    def distanceMatrix(self):
        distanceMatrix = 100*np.zeros((self.num,self.num))
        for i in range(self.num):
            for j in range(self.num):
                distanceMatrix[i][j]=np.sqrt((self.location[i][0]-self.location[j][0])**2+(self.location[i][1]-self.location[j][1])**2)
        return distanceMatrix
    #画出初始点图
    def plotGraph(self):
        location = self.location.T
        plt.scatter(location[0],location[1])
        plt.show()
    # 画出指定路径图
    def plotPath(self,path):
        path_=[]
        for i in path:
            path_.append(self.location[i])
        path_=np.array(path_)
        path_=path_.T
        plt.figure()
        plt.plot(path_[0],path_[1],'-ro')
        plt.plot((path_[0][-1],path_[0][0]),(path_[1][-1],path_[1][0]),'-ro')
        plt.show()
    # 计算某个路径的总长度
    def fitness(self,path):
        result = 0
        for i in range(self.num-1):
            result += self.distance_matrix[path[i]][path[i+1]]
        result += self.distance_matrix[path[-1]][path[0]]
        return result
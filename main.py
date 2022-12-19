from PSO import PSO
from GA import GA

'''
pso = PSO(city_num=30,particle_num=2000,Gbestr=0.2,Pbestr=0.7)
pso.update(100)
pso.plot()
'''

ga = GA(city_num=20,group_num=50,group_n=10,cross_r=0.7,mutation_r=0.2)
ga.update(1000)
ga.plot()
#ga.group.Print()

import torch
import numpy as np
from collections import deque
from optimizer.learnable_optimizer import Learnable_Optimizer
from optimizer.operators import rand_1_single, rand_2_single, rand_to_best_2_single, cur_to_rand_1_single, binomial, \
	clipping


# def init_population_torch(population_size, dim, lb, ub):
# 	population = lb + (ub - lb) * torch.rand(population_size, dim)
# 	return population

def generate_random_int(NP: int, cols: int) -> torch.Tensor:
	r = torch.randint(0, NP, (NP, cols), dtype=torch.long)  # [NP, 3]

	for i in range(NP):
		while r[i, :].eq(i).any():
			r[i, :] = torch.randint(0, NP, (cols,), dtype=torch.long)

	return r


class Surr_RLDE_Optimizer(Learnable_Optimizer):
	def __init__(self, config):
		super().__init__(config)
		# def __init__(self, func, lb, ub, pop_size, dim, mut_way, max_iter, F=0.5, CR_prob=0.5, init_pop=None,
		# 			 device='cuda'):
		# 基础参数设置
		self.config = config
		config.F = 0.5
		config.Cr = 0.7
		config.NP = 100
		self.device = config.device
		self.config = config
		self.F = config.F
		self.Cr = config.Cr
		self.pop_size = config.NP
		self.maxFEs = config.maxFEs
		self.dim = config.dim
		self.ub = config.upperbound
		self.lb = -config.upperbound
		# 种群与适应度
		self.population = None
		self.fitness = None
		self.pop_cur_best = None
		self.fit_cur_best = None
		self.pop_history_best = None
		self.fit_history_best = None
		self.fit_init_best = None
		self.improved_gen = 0

		self.fes = None  # record the number of function evaluations used
		self.cost = None
		self.cur_logpoint = None  # record the current logpoint
		self.log_interval = config.log_interval

		self.is_train = True
	# if self.__cur_checkpoint == 0:
	# 	save_class(self.__config.agent_save_dir, 'checkpoint' + str(self.__cur_checkpoint), self)
	# 	self.__cur_checkpoint += 1

	def get_state(self, problem):
		state = torch.zeros(9)
		# state 1 所有点之间的平均距离
		diff = self.population.unsqueeze(0) - self.population.unsqueeze(1)
		distances = torch.sqrt(torch.sum(diff ** 2, dim=2))  # 距离矩阵
		state[0] = torch.sum(distances) / (self.population.shape[0] * (self.population.shape[0] - 1))

		# state 2 所有点与当前代的最优点的平均距离
		diff = self.population - self.pop_cur_best
		distances = torch.sqrt(torch.sum(diff ** 2, dim=1))
		state[1] = torch.sum(distances) / (self.population.shape[0])

		# state 3 所有点与历史最优点的平均距离
		diff = self.population - self.pop_history_best
		distances = torch.sqrt(torch.sum(diff ** 2, dim=1))
		state[2] = torch.sum(distances) / (self.population.shape[0])

		# state 4 所有y与历史最优的y的距离
		diff = self.fitness - self.fit_history_best
		distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))
		state[3] = torch.sum(distances) / (self.fitness.shape[0])

		# state 5 所有y与当前代最优y的距离
		diff = self.fitness - self.fit_cur_best
		# print

		distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))
		state[3] = torch.sum(distances) / (self.fitness.shape[0])

		# state 6 std(y)
		state[5] = torch.std(self.fitness)

		# state 7 (T - t)/T
		state[6] = (self.maxFEs - self.fes) / self.maxFEs

		# state 8 多少代有提升
		if self.fit_cur_best < self.fit_history_best:
			self.improved_gen = 0
		else:
			self.improved_gen += 1

		state[7] = self.improved_gen

		# state 9 bool型 有提升为1，没有为0
		if self.fit_cur_best < self.fit_history_best:
			state[8] = 1
		else:
			state[8] = 0

		# print(state)
		return state

	def init_population(self, problem):
		self.population = torch.rand(self.pop_size, self.dim) * (problem.ub - problem.lb) + problem.lb
		#(-5,5)
		self.population = self.population.to(self.device)
  
		if self.is_train:
			#训练时使用代理函数，需要对种群进行标准化至(-1,1),已经在problem的eval函数中进行过标准化了
			self.fitness = problem.eval(self.population, is_train=True)

		else:
			#测试时在原始函数，所以种群为(-5,5)
			self.population = self.population.numpy()

			if problem.optimum is None:
				self.fitness = problem.eval(self.population)
			else:
				self.fitness = problem.eval(self.population) - problem.optimum

			self.population = torch.from_numpy(self.population).to(self.device)
			self.fitness = torch.from_numpy(self.fitness).to(self.device)
  
		self.pop_cur_best = self.population[torch.argmin(self.fitness)].clone()
		self.pop_history_best = self.population[torch.argmin(self.fitness)].clone()

		self.fit_init_best = torch.min(self.fitness).clone()
		self.fit_cur_best = torch.min(self.fitness).clone()
		self.fit_history_best = torch.min(self.fitness).clone()

		self.fes = self.pop_size
		self.cost = [self.fit_cur_best.clone().cpu().item()]  # record the best cost of first generation
		self.cur_logpoint = 1  # record the current logpoint
		state = self.get_state(problem)

		return state

	def update(self, action, problem):
		'''
		F:0.1,0.5,0.9
		mutation: rand1,best1,current-rand,current-pbest(p = 10%),currend-best
		共15个动作
		'''

		if action == 0:
			mut_way = 'DE/rand/1'
			self.F = 0.1
		elif action == 1:
			mut_way = 'DE/rand/1'
			self.F = 0.5
		elif action == 2:
			mut_way = 'DE/rand/1'
			self.F = 0.9
		elif action == 3:
			mut_way = 'DE/best/1'
			self.F = 0.1
		elif action == 4:
			mut_way = 'DE/best/1'
			self.F = 0.5
		elif action == 5:
			mut_way = 'DE/best/1'
			self.F = 0.9
		elif action == 6:
			mut_way = 'DE/current-to-rand'
			self.F = 0.1
		elif action == 7:
			mut_way = 'DE/current-to-rand'
			self.F = 0.5
		elif action == 8:
			mut_way = 'DE/current-to-rand'
			self.F = 0.9
		elif action == 9:
			mut_way = 'DE/current-to-pbest'
			self.F = 0.1
		elif action == 10:
			mut_way = 'DE/current-to-pbest'
			self.F = 0.5
		elif action == 11:
			mut_way = 'DE/current-to-pbest'
			self.F = 0.9
		elif action == 12:
			mut_way = 'DE/current-to-best'
			self.F = 0.1
		elif action == 13:
			mut_way = 'DE/current-to-best'
			self.F = 0.5
		elif action == 14:
			mut_way = 'DE/current-to-best'
			self.F = 0.9
		else:
			raise ValueError(f'action error: {action}')
		# 开始进化交叉
		mut_population = self.mutation(mut_way)
		crossover_population = self.crossover(mut_population)

		# 选择
		with torch.no_grad():

			if self.is_train:
				input = crossover_population.to(self.device)
				temp_fit = problem.eval(input,is_train=True).flatten()
			else:
				input = crossover_population.cpu().numpy()
				if problem.optimum is None:
					temp_fit = problem.eval(input)
				else:
					temp_fit = problem.eval(input) - problem.optimum
				temp_fit = torch.from_numpy(temp_fit.flatten())

		for i in range(self.pop_size):
			if temp_fit[i].item() < self.fitness[i].item():
				self.fitness[i] = temp_fit[i]
				self.population[i] = crossover_population[i]
			# print('1')
		# print('111',self.fitness)
		# print('222',temp_fit)

		# 这里的fit_cur_best还没更新，是上一代的最佳
		# print('pre',self.fit_cur_best)
		# print('cur',torch.min(self.fitness).clone())

		reward = self.fit_history_best > torch.min(self.fit_history_best, torch.min(self.fitness).clone())
		# print(reward)

		# 计算当代的最佳
		best_index = torch.argmin(self.fitness)
		# print(best_index)
		self.pop_cur_best = self.population[best_index].clone()
		self.fit_cur_best = self.fitness[best_index].clone()
		# 计算下一个状态
		next_state = self.get_state(problem)

		if self.fit_cur_best < self.fit_history_best:
			self.fit_history_best = self.fit_cur_best.clone()
			self.pop_history_best = self.pop_cur_best.clone()

		is_done = (self.fes >= self.maxFEs)

		self.fes += self.pop_size

		if self.fes >= self.cur_logpoint * self.config.log_interval:
			self.cur_logpoint += 1
			self.cost.append(self.fit_history_best.clone().cpu().item())

		if is_done:
			if len(self.cost) >= self.config.n_logpoint + 1:
				self.cost[-1] = self.fit_history_best.clone().cpu().item()
			else:
				self.cost.append(self.fit_history_best.clone().cpu().item())
		# print(self.pop_cur_best)
		# print(self.cost)

		return next_state, reward.item(), is_done

	def mutation(self, mut_way):
		mut_population = torch.zeros_like(self.population, device=self.device)

		if mut_way == 'DE/rand/1':
			# 使用 generate_random_int 生成随机个体索引
			r = generate_random_int(self.pop_size, 3)  # Shape: [pop_size, 3]
			# print(r)
			# print(r[:, 0], r[:, 1], r[:, 2])
			a = self.population[r[:, 0]]
			b = self.population[r[:, 1]]
			c = self.population[r[:, 2]]
			# print(a, b, c)
			v = a + self.F * (b - c)
			# print(v)
			# print(self.lb,self.ub)
			v = torch.clamp(v, min=self.lb, max=self.ub)
			# print(u)
			mut_population = v

		elif mut_way == 'DE/best/1':
			# 选择 2 个随机个体和最佳个体
			r = generate_random_int(self.pop_size, 2)  # Shape: [pop_size, 2]
			a = self.population[r[:, 0]]
			b = self.population[r[:, 1]]
			v = self.pop_cur_best + self.F * (a - b)
			v = torch.clamp(v, self.lb, self.ub)
			mut_population = v

		elif mut_way == 'DE/current-to-rand':
			# 选择 3 个随机个体并计算突变
			r = generate_random_int(self.pop_size, 3)  # Shape: [pop_size, 3]
			a = self.population[r[:, 0]]
			b = self.population[r[:, 1]]
			c = self.population[r[:, 2]]
			v = self.population + self.F * (a - self.population) + self.F * (b - c)
			v = torch.clamp(v, self.lb, self.ub)
			mut_population = v

		elif mut_way == 'DE/current-to-pbest':
			# 获取 pbest 个体并进行突变
			p = 0.1
			p_num = max(1, int(p * self.pop_size))
			sorted_indices = torch.argsort(self.fitness.clone().flatten())
			pbest_indices = sorted_indices[:p_num]
			# 选择 2 个随机个体和一个 pbest 个体
			r = generate_random_int(self.pop_size, 2)  # Shape: [pop_size, 2]

			a = self.population[r[:, 0]]
			b = self.population[r[:, 1]]

			pbest_index = pbest_indices[torch.randint(0, p_num, (self.pop_size,))]
			pbest = self.population[pbest_index]

			v = self.population + self.F * (pbest - self.population) + self.F * (a - b)
			v = torch.clamp(v, self.lb, self.ub)
			mut_population = v

		elif mut_way == 'DE/current-to-best':
			# 选择 4 个随机个体和最佳个体进行突变
			r = generate_random_int(self.pop_size, 4)  # Shape: [pop_size, 4]
			a = self.population[r[:, 0]]
			b = self.population[r[:, 1]]
			c = self.population[r[:, 2]]
			d = self.population[r[:, 3]]
			v = self.population + self.F * (self.pop_cur_best - self.population) + self.F * (a - b) + self.F * (c - d)
			v = torch.clamp(v, self.lb, self.ub)
			mut_population = v

		else:
			print('请选择正确的变异策略')

		mut_population = torch.tensor(mut_population, device=self.device)
		return mut_population

	def crossover(self, mut_population):
		crossover_population = self.population.clone()

		# print('交叉前',crossover_population)
		for i in range(self.pop_size):

			select_dim = torch.randint(0, self.dim, (1,))
			for j in range(self.dim):
				if torch.rand(1) < self.Cr or j == select_dim:
					crossover_population[i][j] = mut_population[i][j]
		# print('交叉后',crossover_population)
		return crossover_population

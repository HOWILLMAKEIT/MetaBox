import torch.nn as nn

# MLP
class MLP(nn.Module):
	def __init__(self, input_dim):
		super(MLP, self).__init__()
		self.ln1 = nn.Linear(input_dim, 32)
		self.ln2 = nn.Linear(32, 64)
		self.ln3 = nn.Linear(64, 32)
		self.ln4 = nn.Linear(32, 1)
		self.relu1 = nn.ReLU()
		self.relu2 = nn.ReLU()
		self.relu3 = nn.ReLU()

	def forward(self, x):
		x = self.ln1(x)
		x = self.relu1(x)
		x = self.ln2(x)
		x = self.relu2(x)
		x = self.ln3(x)
		x = self.relu3(x)
		x = self.ln4(x)
		return x
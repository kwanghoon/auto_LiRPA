"""
A toy example for bounding neural network outputs under input perturbations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

class simple_model(torch.nn.Module):
    """
    A very simple 2-layer neural network for demonstration.
    """
    def __init__(self):
        super().__init__()
        # Weights of linear layers.
        self.w1 = torch.tensor([[1., -1.], [2., -1.]])
        self.w2 = torch.tensor([[1., -1.]])

    def forward(self, x):
        # Linear layer.
        z1 = x.matmul(self.w1.t())
        # Relu layer.
        hz1 = torch.nn.functional.relu(z1)
        # Linear layer.
        z2 = hz1.matmul(self.w2.t())
        return z2

class WirelessNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        
        # layer 1
        self.layer1 = nn.Linear(256, 491, bias=False)
        self.bn1 = nn.BatchNorm1d(491)
        
        # layer 2
        self.layer2 = nn.Linear(491, 491, bias=False)
        self.bn2 = nn.BatchNorm1d(491)
        
        # layer 3
        self.layer3 = nn.Linear(491, 16, bias=False)
        self.bn3 = nn.BatchNorm1d(16)

        # initialize weights
        nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer3.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        # layer 1
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # layer 2
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # layer 3
        x = self.layer3(x)
        x = self.bn3(x)
        x = torch.sigmoid(x)
        
        return x


model = WirelessNeuralNetwork()


'''
model = simple_model()

# Input x.
x = torch.tensor([[1., 1.]])
# Lowe and upper bounds of x.
lower = torch.tensor([[-1., -2.]])
upper = torch.tensor([[2., 1.]])

# Wrap model with auto_LiRPA for bound computation.
# The second parameter is for constructing the trace of the computational graph,
# and its content is not important.
lirpa_model = BoundedModule(model, torch.empty_like(x))
pred = lirpa_model(x)
print(f'Model prediction: {pred.item()}')

# Compute bounds using LiRPA using the given lower and upper bounds.
norm = float("inf")
ptb = PerturbationLpNorm(norm = norm, x_L=lower, x_U=upper)
bounded_x = BoundedTensor(x, ptb)

# Compute bounds.
lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='IBP')
print(f'IBP bounds: lower={lb.item()}, upper={ub.item()}')
lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
print(f'CROWN bounds: lower={lb.item()}, upper={ub.item()}')

# Getting the linear bound coefficients (A matrix).
required_A = defaultdict(set)
required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
lb, ub, A = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN', return_A=True, needed_A_dict=required_A)
print('CROWN linear (symbolic) bounds: lA x + lbias <= f(x) <= uA x + ubias, where')
print(A[lirpa_model.output_name[0]][lirpa_model.input_name[0]])

# Opimized bounds, which is tighter.
lb, ub, A = lirpa_model.compute_bounds(x=(bounded_x,), method='alpha-CROWN', return_A=True, needed_A_dict=required_A)
print(f'alpha-CROWN bounds: lower={lb.item()}, upper={ub.item()}')
print('alpha-CROWN linear (symbolic) bounds: lA x + lbias <= f(x) <= uA x + ubias, where')
print(A[lirpa_model.output_name[0]][lirpa_model.input_name[0]])
'''

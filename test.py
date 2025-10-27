import torch
from torch.distributions import Categorical

p0 = Categorical(probs=torch.tensor([0.25, 0.25, 0.25, 0.25]))
p1 = Categorical(probs=torch.tensor([0.3, 0.3, 0.2, 0.2]))
p2 = Categorical(probs=torch.tensor([0.4, 0.2, 0.2, 0.2]))
p3 = Categorical(probs=torch.tensor([0.5, 0.1, 0.2, 0.2]))
p4 = Categorical(probs=torch.tensor([0.6, 0.1, 0.1, 0.2]))
p5 = Categorical(probs=torch.tensor([0.7, 0.1, 0.1, 0.1]))
p6 = Categorical(probs=torch.tensor([0.8, 0.05, 0.05, 0.1]))
p7 = Categorical(probs=torch.tensor([0.9, 0.05, 0.025, 0.025]))
p8 = Categorical(probs=torch.tensor([0.95, 0.0125, 0.0125, 0.025]))

policies = [
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
]

log_probs = [p.log_prob(torch.tensor(0)) for p in policies]

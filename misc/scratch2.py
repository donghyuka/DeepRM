import torch

t = torch.zeros(5,50)
t[0,10:20] = 1
t[1,20:30] = 1
t[2,5:25] = 1
t[3,35:40] = 1
t[4,10:25] = 1

print(t)

t = t.to(1)

x = torch.arange(t.shape[1], device = t.device).unsqueeze(0).repeat(t.shape[0],1)
t_sigma = t.sum(dim = 1, keepdim=True) / 2
t_mu = t_sigma + (x * t * torch.cat([torch.zeros(t.shape[0],1, device = t.device), t[:,:-1]], dim = 1).logical_not()).sum(dim = 1, keepdim=True)

print(t_mu)
print(t_sigma)

t = torch.exp(-((x - t_mu) ** 2) / (2 * t_sigma ** 2))

print(t)
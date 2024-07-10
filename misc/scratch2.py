import torch
t = torch.rand((4,100))
print(t[0])

t = t.unfold(1, 10, 5)
print(t[0])
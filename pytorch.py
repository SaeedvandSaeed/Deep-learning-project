from __future__ import print_function
import torch
# x = torch.empty(5, 3)


#x = torch.tensor([5.5, 3])


# x = torch.zeros(5, 3)

# y = torch.rand(5, 3)
# print(x + y)

x = torch.rand(4, 3)
# print(x)
# print()

# print(x[:, 1])
# print()
# print(x[0, 1:])

# y = x.view(12)
# y = x.view(6, -1)
# print(y)
# print(y.size())

# y = x.view(-1, 6)
y = x.view(1, -1)

print(x)
print()
print(y)
print(y.size())

nn = x.numpy()  # converting tensor to numpy
bb = torch.from_numpy(nn)  # converting numpy to tensor

print(nn)
# print(bb)
# print(x.size())

# print(x)

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    # or just use strings ``.to("cuda")``
    x = x.to(device)
    z = x + y
    print(z)
    # ``.to`` can also change dtype together!
    print(z.to("cpu", torch.double))

import torch

def make_func_and_grad(loss_func, shape, device, dtype=None):
    def func_and_grad(x):
        x_tensor = torch.tensor(x, dtype=dtype)
        x_tensor = x_tensor.view(*shape)
        x_tensor = x_tensor.to(device)
        x_tensor.requires_grad = True
        
        loss = loss_func(x_tensor)
        loss.backward()
        return loss.item(), x_tensor.grad.cpu().numpy().flatten()

    return func_and_grad

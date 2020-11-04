import torch

def make_func_and_grad(loss_func, shape, device, dtype=None):
    """Wraps loss_func to take and return numpy 1D arrays, for interfacing PyTorch and copt.
    
    Args:
      loss_func: callable
        PyTorch callable, taking a torch.Tensor a input, and returning a scalar
    
      shape: tuple(*int)
        shape of the optimization variable, as input to loss_func
        
      device: torch.Device
        device on which to send the optimization variable
    
      dtype: dtype
        data type for the torch.Tensor holding the optimization variable 
    
    Returns:
      f_grad: callable
        function taking a 1D numpy array as input and returning (loss_val, grad_val): (float, array).
    """
    def func_and_grad(x, return_gradient=True):
        x_tensor = torch.tensor(x, dtype=dtype)
        x_tensor = x_tensor.view(*shape)
        x_tensor = x_tensor.to(device)
        x_tensor.requires_grad = True
        
        loss = loss_func(x_tensor)
        loss.backward()
        if return_gradient:
            return loss.item(), x_tensor.grad.cpu().numpy().flatten()

        return loss.item()
    return func_and_grad

# TODO: write generic function wrapping copt optimizers for taking pytorch input, 
# returning pytorch output for use of copt in a PyTorch pipeline
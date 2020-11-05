from functools import partial
import numpy as np
import torch

from tqdm import tqdm

import copt
from copt.utils_pytorch import make_func_and_grad

from robustbench.data import load_cifar10
from robustbench.utils import load_model


n_examples = 6000
data_batch, target_batch = load_cifar10(n_examples=n_examples, data_dir='~/datasets')

model_name = "Engstrom2019Robustness"
model = load_model(model_name)  # loads a standard trained model
criterion = torch.nn.CrossEntropyLoss()
# Define the constraint set
alpha = 0.5
constraint = copt.constraint.L2Ball(alpha)

n_correct = 0
n_correct_adv = 0


# Define the loss function to be minimized, using Pytorch
def loss_fun(delta, data):
    adv_input = data + delta
    return -criterion(model(adv_input), target)


print(f"Evaluating model {model_name}, on L{constraint.p}Ball({alpha}).")

for k, (data, target) in tqdm(enumerate(zip(data_batch, target_batch))):
    data, target = data.unsqueeze(0), target.unsqueeze(0)

    loss_fun_data = partial(loss_fun, data=data)
    # Change the function to f_grad: returns loss_val, grad in flattened, numpy array
    f_grad = make_func_and_grad(loss_fun_data, data.shape, data.device, dtype=data.dtype)

    img_np = data.cpu().numpy().squeeze().flatten()

    def image_constraint_prox(delta, step_size=None):
        """Projects perturbation delta so that x + delta is in the set of images,
        i.e. the (0, 1) range."""
        adv_img_np = img_np + delta
        delta = adv_img_np.clip(0, 1) - img_np
        return delta

    delta0 = np.zeros(data.shape, dtype=float).flatten()

    callback = copt.utils.Trace(lambda delta: f_grad(delta)[0])

    sol = copt.minimize_three_split(f_grad, delta0, constraint.prox,
                                    image_constraint_prox, callback=callback,
                                    max_iter=25
                                    )
    label = torch.argmax(model(data), dim=-1)

    pert_tensor = torch.tensor(sol.x, dtype=data.dtype).to(data.device)
    pert_tensor = pert_tensor.reshape(data.shape)
    adv_label = torch.argmax(model(torch.clamp(data + pert_tensor, 0., 1.)), dim=-1)

    n_correct += (label == target).item()
    n_correct_adv += (adv_label == target).item()

    if k % 100 == 1:
        curr_acc = n_correct / (k + 1)
        curr_acc_adv = n_correct_adv / (k + 1)
        print(f"\nAccuracy so far: {curr_acc:.3f}")
        print(f"Robust Accuracy so far: {curr_acc_adv:.3f}")

accuracy = n_correct / n_examples
accuracy_adv = n_correct_adv / n_examples

print(f"Accuracy: {accuracy:.3f}")
print(f"Robust accuracy: {accuracy_adv:.3f}")

import numpy as np
import torch
import copt
from copt.utils_pytorch import make_func_and_grad

from robustbench.data import load_cifar10
from robustbench.utils import load_model

import matplotlib.pyplot as plt

n_examples = 100
data, target = load_cifar10(n_examples=n_examples, data_dir='~/datasets')

np.random.seed(1)
idx = np.random.randint(n_examples)
data, target = data[idx].unsqueeze(0), target[idx].unsqueeze(0)

model = load_model("Standard")  # loads a standard trained model

criterion = torch.nn.CrossEntropyLoss()


# Define the loss function to be minimized, using Pytorch
def loss_fun(delta):
    adv_input = data + delta
    return -criterion(model(adv_input), target)

# Change the function to f_grad: returns loss_val, grad in flattened, numpy array
f_grad = make_func_and_grad(loss_fun, data.shape, data.device, dtype=data.dtype)

# Define the constraint set
alpha = 1. 
constraint = copt.constraint.L1Ball(alpha)

delta0 = np.zeros(data.shape, dtype=float).flatten()
delta0[0] += alpha

callback = copt.utils.Trace(lambda delta: f_grad(delta)[0])

# TODO: wrap this in a function with Pytorch input and outputs
sol = copt.minimize_frank_wolfe(f_grad, delta0,
                                constraint.lmo,
                                # constraint.lmo_pairwise,
                                max_iter=300,
                                x0_rep=(1., 0),
                                # variant='pairwise',
                                callback=callback)

fig, ax = plt.subplots()
ax.plot([-loss_val for loss_val in callback.trace_fx], lw=3)
ax.set_yscale("log")
ax.set_xlabel("# Iterations")
ax.set_ylabel("Objective value")
ax.set_title("")
ax.grid()
plt.savefig("figures/adversarial_loss.png")
plt.show()

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

img = data.cpu().numpy().squeeze()
perturbation = sol.x.reshape(img.shape)
adv_img = img + perturbation

img = img.transpose(1, 2, 0)
perturbation = perturbation.transpose(1, 2, 0)
adv_img = adv_img.transpose(1, 2, 0)
adv_img = np.clip(adv_img, 0, 1)

perturbation = adv_img - img

fig, axes = plt.subplots(ncols=3)
img_ax, pert_ax, adv_img_ax =  axes

label = torch.argmax(model(data))

pert_tensor = torch.tensor(sol.x, dtype=data.dtype).to(data.device)
pert_tensor = pert_tensor.reshape(data.shape)
adv_label = torch.argmax(model(torch.clamp(data + pert_tensor, 0., 1.)))

img_ax.set_title(f"Original image: {classes[label]}")
img_ax.imshow(img)

pert_ax.set_title("Perturbation")
pert_ax.imshow(perturbation)

adv_img_ax.set_title(f"Perturbed image: {classes[adv_label]}")
adv_img_ax.imshow(adv_img)

plt.savefig("figures/adv_example.png")

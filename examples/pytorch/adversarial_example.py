import numpy as np
import torch
import copt
from copt.utils_pytorch import make_func_and_grad

from robustbench.data import load_cifar10
from robustbench.utils import load_model

import matplotlib.pyplot as plt

n_examples = 20
data_batch, target_batch = load_cifar10(n_examples=n_examples, data_dir='~/datasets')

for data, target in zip(data_batch, target_batch):
    data, target = data.unsqueeze(0), target.unsqueeze(0)

    model = load_model("Engstrom2019Robustness")  # loads a standard trained model

    criterion = torch.nn.CrossEntropyLoss()


    # Define the loss function to be minimized, using Pytorch
    def loss_fun(delta):
        adv_input = data + delta
        return -criterion(model(adv_input), target)

    # Change the function to f_grad: returns loss_val, grad in flattened, numpy array
    f_grad = make_func_and_grad(loss_fun, data.shape, data.device, dtype=data.dtype)

    # Define the constraint set
    alpha = 10. 
    constraint = copt.constraint.L1Ball(alpha)

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
                                    max_iter=50
                                    )

    fig, ax = plt.subplots()
    ax.plot([-loss_val for loss_val in callback.trace_fx], lw=3)
    ax.set_yscale("log")
    ax.set_xlabel("# Iterations")
    ax.set_ylabel("Objective value")
    ax.grid()

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

    # Project back so that 0 <= x + delta <= 1
    adv_img = np.clip(adv_img, 0, 1)
    perturbation = adv_img - img

    fig, axes = plt.subplots(ncols=3)
    img_ax, pert_ax, adv_img_ax =  axes

    output = torch.nn.functional.softmax(model(data), dim=-1)
    label = torch.argmax(output)

    pert_tensor = torch.tensor(sol.x, dtype=data.dtype).to(data.device)
    pert_tensor = pert_tensor.reshape(data.shape)
    adv_output = torch.nn.functional.softmax(model(torch.clamp(data + pert_tensor, 0., 1.)), dim=-1)
    adv_label = torch.argmax(adv_output)

    img_ax.set_title(f"Original image: {classes[label]}, p={output[:, label].item():.2f}")
    img_ax.imshow(img)

    pert_ax.set_title("Perturbation")
    pert_ax.imshow(abs(perturbation))

    adv_img_ax.set_title(f"Perturbed image: {classes[adv_label]}, p={adv_output[:, adv_label].item():.2f}")
    adv_img_ax.imshow(adv_img)
    plt.tight_layout()

    plt.show()

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import utils

matplotlib.use('Agg')


def get_gen_output_figure_1d(challenge, real_response, gen_response, challenge_shape, response_shape):
    cols = 4
    rows = min(6, challenge.shape[0])
    fig, axs = plt.subplots(rows, cols, figsize=(16, 20))

    for row in range(rows):
        real = real_response[row]
        gen = gen_response[row]
        pc = utils.calc_pear_coeff(real, gen).item()
        rel_error = (np.linalg.norm(real.cpu().numpy() - gen.cpu().numpy()) / np.linalg.norm(real.cpu().numpy())).item()
        curr_challenge = challenge[row]

        show_img_in_ax(axs[row, 0], curr_challenge.reshape(challenge_shape), challenge_shape)
        show_img_in_ax(axs[row, 1], real, response_shape, snakify=True)
        show_img_in_ax(axs[row, 2], gen, response_shape, snakify=True, title=f"PC: {pc:.3f}, RE: {rel_error:.3f}")
        show_difference_map_in_ax_1d(axs[row, 3], real, gen, response_shape)
    return fig


def get_gen_output_figure_complex_1d(challenge, real_response, gen_response, challenge_shape, response_shape):
    real_response = (real_response - torch.min(real_response)) / (torch.max(real_response) - torch.min(real_response))
    gen_response = (gen_response - torch.min(gen_response)) / (torch.max(gen_response) - torch.min(gen_response))

    cols = 7
    rows = min(6, challenge.shape[0])
    fig, axs = plt.subplots(rows, cols, figsize=(16, 20))

    for row in range(rows):
        real = real_response[row]
        gen = gen_response[row]
        pc1 = utils.calc_pear_coeff(real[0], gen[0]).item()
        pc2 = utils.calc_pear_coeff(real[1], gen[1]).item()
        curr_challenge = challenge[row]

        show_img_in_ax(axs[row, 0], curr_challenge.reshape(challenge_shape), challenge_shape)
        show_img_in_ax(axs[row, 1], real[0], response_shape, snakify=True)
        show_img_in_ax(axs[row, 2], gen[0], response_shape, snakify=True, title=f"{pc1:.3f}")
        show_difference_map_in_ax_1d(axs[row, 3], real[0], gen[0], 23)
        show_img_in_ax(axs[row, 4], real[1], response_shape, snakify=True)
        show_img_in_ax(axs[row, 5], gen[1], response_shape, snakify=True, title=f"{pc2:.3f}")
        show_difference_map_in_ax_1d(axs[row, 6], real[1], gen[1], 23)
    return fig


def get_gen_output_figure_2d(challenge, real_response, gen_response, challenge_shape, response_shape):
    cols = 4
    rows = min(6, challenge.shape[0])
    fig, axs = plt.subplots(rows, cols, figsize=(16, 20))

    for row in range(rows):
        real = real_response[row]
        gen = gen_response[row]
        pc = utils.calc_pear_coeff(real, gen).item()
        rel_error = (np.linalg.norm(real.cpu().numpy() - gen.cpu().numpy()) / np.linalg.norm(real.cpu().numpy())).item()
        curr_challenge = challenge[row]

        show_img_in_ax(axs[row, 0], curr_challenge, challenge_shape)
        show_img_in_ax(axs[row, 1], real, response_shape)
        show_img_in_ax(axs[row, 2], gen, response_shape, title=f"PC: {pc:.3f}, RE: {rel_error:.3f}")
        show_difference_map_in_ax_2d(axs[row, 3], real, gen)
    return fig


def pearsonr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.linalg.norm(xm, 2) * torch.linalg.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


def show_img_in_ax(ax, img, img_size, snakify=False, title=""):
    ax.axis("off")
    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')

    if title:
        ax.set_title(title)

    img = img.cpu().numpy().squeeze()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    if snakify:
        img_copy = img.copy()
        img_copy.resize(img_size, refcheck=False)
        img_copy = np.reshape(img_copy, img_size)
        img_copy[1::2] = img_copy[1::2, ::-1]
        img = img_copy
    else:
        img = np.reshape(img, img_size)

    ax.imshow(img, cmap="gray", vmin=0, vmax=1)


def show_difference_map_in_ax_1d(ax, real_response, gen_response, img_size):
    ax.axis("off")

    real_response = torch.squeeze(real_response).cpu().numpy()
    gen_response = torch.squeeze(gen_response).cpu().numpy()

    difference_map = np.absolute((real_response - gen_response))
    difference_map_copy = difference_map.copy()
    difference_map_copy.resize(img_size, refcheck=False)
    difference_map_copy = np.reshape(difference_map_copy, img_size)

    difference_map_copy[1::2] = difference_map_copy[1::2, ::-1]
    ax.set_title(f"Abs Diff.")
    ax.imshow(difference_map_copy, cmap="gray", vmin=0, vmax=1)


def show_difference_map_in_ax_2d(ax, real_response, gen_response):
    ax.axis("off")

    real_response = torch.squeeze(real_response).cpu().numpy()
    real_response = (real_response - np.min(real_response)) / (np.max(real_response) - np.min(real_response))

    gen_response = torch.squeeze(gen_response).cpu().numpy()
    gen_response = (gen_response - np.min(gen_response)) / (np.max(gen_response) - np.min(gen_response))

    difference_map = np.absolute((real_response - gen_response))

    ax.set_title(f"Abs Diff.")
    ax.imshow(difference_map, cmap="gray", vmin=0, vmax=1)


def plot_grad_flow(named_parameters, axis):
    '''
    Plots the gradients flowing through different layers in the net during training. Assumes that a figure was
    initiated beforehand.
    '''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())

    axis.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    axis.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    axis.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    axis.set_xticks(range(0, len(ave_grads), 1))
    axis.set_xticklabels(layers, rotation="vertical")
    axis.set_xlim(left=0, right=len(ave_grads))
    axis.set_ylim(bottom=-0.001, top=0.2)
    axis.set_xlabel("Layers")
    axis.set_ylabel("Average gradient")
    axis.grid(True)
    axis.legend([Line2D([0], [0], color="c", lw=4),
                 Line2D([0], [0], color="b", lw=4),
                 Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def create_gradient_figure(name):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_title(f"{name} Gradient flow")
    return fig, ax

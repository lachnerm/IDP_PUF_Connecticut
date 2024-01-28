import numpy as np
import torch
from torch import nn


def calc_pc_tensor(real_response, gen_response):
    return [pearsonr(r.flatten(), g.flatten()).item() for (r, g) in
            zip(real_response, gen_response)]


def calc_rel_error_tensor(real_response, gen_response):
    return [
        (np.linalg.norm(real.cpu() - gen.cpu()) / np.linalg.norm(real.cpu()))
        for real, gen in
        zip(real_response, gen_response)]


def calc_diff_map_mean(real_response, gen_response):
    real_response = real_response * 0.5 + 0.5
    gen_response = gen_response * 0.5 + 0.5

    real_response = real_response.cpu().numpy()
    gen_response = gen_response.cpu().numpy()

    diff_map = np.absolute((real_response - gen_response))
    return np.mean(diff_map) * 255


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def calc_pear_coeff(real_response, gen_response):
    y = real_response
    x = gen_response

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    cost = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return cost


def pearsonr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


def pearsonr_loss(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    return torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))


def pc_loss_func(*args):
    return 1 - pearsonr_loss(*args)


def store_preds(challenges, responses, store_path):
    with open(f"{store_path}/preds.npy", "wb") as cf:
        challenges = challenges.squeeze().cpu().numpy()
        responses = responses.squeeze().cpu().numpy()
        np.savez(cf, challenges=challenges, responses=responses)

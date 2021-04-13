import numpy as np
import torch.nn as nn
import math
import modules.config as config


def ramp_up(epoch, rump_up_epochs=config.rampup_length):
    """
    Ramps up the value of the learning rate
    in the first 80 epochs
    :param epoch: {int} current epoch
    :param rump_up_epochs:  {int} rump up period length
    :return: {float} ramp up value
    """
    if epoch < rump_up_epochs:
        T = epoch / rump_up_epochs
        return np.exp(-5 * (1 - T) ** 2)
    else:
        return 1.0


def ramp_down(epoch, num_epochs=config.num_epochs):
    """
    Ramps down the value of the learning rate and adam's beta
    in the last 50 epochs
    :param epoch: {int} current epoch
    :param num_epochs: {int} total number of epochs
    :return: {float} ram down value
    """

    epoch_with_max_rampdown = 50

    if epoch >= (num_epochs - epoch_with_max_rampdown):
        ep = (epoch - (num_epochs - epoch_with_max_rampdown)) * 0.5
        return math.exp(-(ep * ep) / epoch_with_max_rampdown)
    else:
        return 1.0


def weights_init(m):
    """
    :param m: Kaiming uniform weight initialization
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)

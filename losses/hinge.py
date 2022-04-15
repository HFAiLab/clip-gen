import torch
import torch.nn.functional as F


def hinge_g_loss(logits_fake):
    """
    GAN generator loss. In this stage, we train Encoder + Generator and freeze Discriminator

    Args:
        logits_fake (Tensor): output of (x -> E -> G -> D), size (N, 1, h, w)

    Returns:
        loss (Tensor): loss value
    """
    loss = -torch.mean(logits_fake)
    return loss


def hinge_d_loss(logits_real, logits_fake):
    """
    GAN discriminator loss. In this stage, we train Discrinimator and freeze Encoder + Generator

    Args:
        logits_real (Tensor): output of (x -> D), size (N, 1, h, w)
        logits_fake (Tensor): output of (x -> E -> G -> D), size (N, 1, h, w)

    Returns:
        loss (Tensor): loss value
    """

    loss_real = torch.mean(F.relu(1. - logits_real))  # logits_real -> +1
    loss_fake = torch.mean(F.relu(1. + logits_fake))  # logits_fake -> -1
    loss = 0.5 * (loss_real + loss_fake)
    return loss

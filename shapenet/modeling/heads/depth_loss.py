import torch
import torch.nn.functional as F


def interpolate_multi_view_tensor(tensor, size):
    """(B, V, H1, W1) -> (B, V, H2, W2)
    """
    B, V = tensor.shape[:2]
    # (B, V, 1, H, W)
    tensor = tensor.unsqueeze(2)
    # (B*V, 1, H, W)
    tensor = tensor.view(-1, *(tensor.shape[2:]))
    # (B, V, , H, W)
    return F.interpolate(tensor, size, mode="nearest").view(B, V, *size)


def adaptive_berhu_loss(depth_gt, depth_est, mask, threshold=0.2):
    """
    Args:
    - depth_gt, depth_est, mask: tensor of shape (B, V, H, W)
    """
    if depth_est is None:
        return torch.tensor(0.0).type(depth_gt.dtype).to(depth_gt.device)
    if depth_gt is None:
        return torch.tensor(0.0).type(depth_est.dtype).to(depth_est.device)

    depth_gt = interpolate_multi_view_tensor(depth_gt, depth_est.shape[-2:])
    mask = interpolate_multi_view_tensor(mask, depth_est.shape[-2:])

    mask = mask.type(depth_gt.dtype).to(depth_gt.device)
    diff = torch.abs(depth_gt * mask - depth_est * mask)
    delta = threshold * torch.max(diff).item()

    l1_part = -F.threshold(-diff, -delta, 0.)
    l2_part = F.threshold(diff**2 - delta**2, 0., -delta**2.) + delta**2
    l2_part = l2_part / (2.*delta)

    loss = l1_part + l2_part
    loss = torch.mean(loss)
    return loss


def adaptive_huber_loss(input, target, beta=1./9, reduction='mean'):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if reduction == 'mean':
        return loss.mean()
    return loss.sum()


def huber_loss(x, y):
    return F.smooth_l1_loss(x, y, reduction='mean')

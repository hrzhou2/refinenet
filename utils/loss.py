
import numpy as np
import torch
import torch.nn.functional as F


def angle_degrees(v1, v2):
    '''
    Inputs:
        v1: (n,3), v2: (n,3)
    Returns:
        degrees: (n,)
    '''
    # lower precision loss
    v1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    v2 = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
    error = np.sum((v1-v2)**2, axis=1)
    angle = np.arccos((2-error)/2)

    return np.degrees(angle)


def cos_angle(v1, v2):

    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)


def compute_loss(output, target, loss_type, normalize=False):
    '''
    Get loss for predicted normals, without reorientation
    Inputs:
        output: (n, 3)
        target: (n, 3)
    '''

    loss = 0

    if normalize:
        output = F.normalize(output, dim=1)
        target = F.normalize(target, dim=1)

    if loss_type == 'mse_loss':
        loss += F.mse_loss(output, target)
    elif loss_type == 'ms_euclidean':
        loss += torch.min((output-target).pow(2).sum(1), (output+target).pow(2).sum(1)).mean()
    elif loss_type == 'ms_oneminuscos':
        loss += (1-torch.abs(cos_angle(output, target))).pow(2).mean()
    elif loss_type == 'l1_loss':
        loss += F.l1_loss(output, target)
    else:
        raise ValueError('Unsupported loss type: {}'.format(loss_type))

    return loss
    

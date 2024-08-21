import torch
import numpy as np
from torch import Tensor, LongTensor
from scipy.optimize import linear_sum_assignment
from typing import Tuple
import torch.nn.functional as F

def cosine_similarity(a: Tensor, b: Tensor, eps: float = 1e-6):
    """Computes the cosine similarity between two tensors.
    Args:
        a (Tensor): Tensor with shape (batch size, N_a, D).
        b (Tensor): Tensor with shape (batch size, N_b, D).
        eps (float): Small constant for numerical stability.
    Returns:
        The (batched) cosine similarity between `a` and `b`, with shape (batch size, N_a, N_b).
    """
    dot_products = torch.matmul(a, torch.swapaxes(b, 1, 2))
    norm_a = (a * a).sum(dim=2).sqrt().unsqueeze(2)
    norm_b = (b * b).sum(dim=2).sqrt().unsqueeze(1)
    return dot_products / (torch.matmul(norm_a, norm_b) + eps)

def cosine_distance(a: Tensor, b: Tensor, eps: float = 1e-6):
    """Computes the cosine distance between two tensors, as 1 - cosine_similarity.
    Args:
        a (Tensor): Tensor with shape (batch size, N_a, D).
        b (Tensor): Tensor with shape (batch size, N_b, D).
        eps (float): Small constant for numerical stability.
    Returns:
        The (batched) cosine distance between `a` and `b`, with shape (batch size, N_a, N_b).
    """
    return 1 - cosine_similarity(a, b, eps)

def get_mask_cosine_distance(true_mask: Tensor, pred_mask: Tensor):
    """Computes the cosine distance between the true and predicted masks.
    Args:
        true_mask (Tensor): Tensor of shape (batch size, num objects, 1, H, W).
        pred_mask (Tensor): Tensor of shape (batch size, num slots, 1, H, W).
    Returns:
        The (batched) cosine similarity between the true and predicted masks, with
        shape (batch size, num objects, num slots).
    """
    return cosine_distance(true_mask.flatten(2).detach(), pred_mask.flatten(2).detach())

def hungarian_algorithm(cost_matrix: Tensor) -> Tuple[Tensor, LongTensor]:
    """Batch-applies the hungarian algorithm to find a matching that minimizes the overall cost.
    Returns the matching indices as a LongTensor with shape (batch size, 2, min(num objects, num slots)).
    The first column is the row indices (the indices of the true objects) while the second
    column is the column indices (the indices of the slots). The row indices are always
    in ascending order, while the column indices are not necessarily.
    The outputs are on the same device as `cost_matrix` but gradients are detached.
    A small example:
                | 4, 1, 3 |
                | 2, 0, 5 |
                | 3, 2, 2 |
                | 4, 0, 6 |
    would result in selecting elements (1,0), (2,2) and (3,1). Therefore, the row
    indices will be [1,2,3] and the column indices will be [0,2,1].
    Args:
        cost_matrix: Tensor of shape (batch size, num objects, num slots).
    Returns:
        A tuple containing:
            - a Tensor with shape (batch size, min(num objects, num slots)) with the
              costs of the matches.
            - a LongTensor with shape (batch size, 2, min(num objects, num slots))
              containing the indices for the resulting matching.
    """
    # List of tuples of size 2 containing flat arrays
    indices = list(map(linear_sum_assignment, cost_matrix.cpu().detach().numpy()))
    indices = torch.LongTensor(np.array(indices))
    smallest_cost_matrix = torch.stack(
        [
            cost_matrix[i][indices[i, 0], indices[i, 1]]
            for i in range(cost_matrix.shape[0])
        ]
    )
    device = cost_matrix.device
    return smallest_cost_matrix.to(device), indices.to(device)

# true_mask = None # from dataset
# attns = None # from model
# max_num_obj = None # from dataset
# attns = F.interpolate(attns, size=true_mask.shape[-1], mode='bilinear')
# true_mask_one_hot = F.one_hot(true_mask.long(), num_classes=max_num_obj + 1).float()
# miou_list_no_bg = []
# mbo_list_no_bg = []
# cost_matrix = get_mask_cosine_distance(true_mask_one_hot[..., None, :, :], attns_one_hot[..., None, :, :]) # attns or attns_one_hot
# # if background is included in visibility and first object in other labels is background: clevr
# selected_objects = labels['visibility']
# selected_objects[:, 0] = 0
# if len(selected_objects.shape) == 2:
#     selected_objects = selected_objects[:, :, None]
# obj_idx_adjustment = 0
# # # if background is not included in visibility and first object in other labels is background: movi series or clevrtex
# # selected_objects = \
# #     torch.cat([torch.zeros_like(labels['visibility'][:, 0:1]).to(labels['visibility'].device),
# #                 labels['visibility'] > 0], dim=1)[..., None]
# # obj_idx_adjustment = 1

# cost_matrix = cost_matrix * selected_objects + 100000 * (1 - selected_objects)
# _, indices = hungarian_algorithm(cost_matrix)

# for idx_in_batch, num_o in enumerate(labels['num_obj']):
#     for gt_idx, pred_idx in zip(indices[idx_in_batch][0], indices[idx_in_batch][1]):
#         if selected_objects[idx_in_batch, ..., 0][
#             gt_idx] == 0:  # no gt_idx - 1 here because we added the background to the beginning
#             continue

#         gt_map = (true_mask[idx_in_batch] == gt_idx)
#         pred_map = (attn_argmax[idx_in_batch] == pred_idx)
#         miou_list_no_bg.append((gt_map & pred_map).sum() / (gt_map | pred_map).sum())

#         iou_all = (gt_map[None] & attns_one_hot[idx_in_batch].bool()).sum((-2, -1)) / \
#                     (gt_map[None] | attns_one_hot[idx_in_batch].bool()).sum((-2, -1))
#         mbo_list_no_bg.append(iou_all.max())



def compute_matching(true_mask, pred_mask, visibility):

    # input to cost_matrix should be one hot
    cost_matrix = get_mask_cosine_distance(true_mask, pred_mask)

    # exclude non visibile objects
    visibility=visibility.cpu()
    cost_matrix = cost_matrix * visibility + 1e7 * (1 - visibility) 
    _, indices = hungarian_algorithm(cost_matrix)

    return indices

def compute_iou(true_mask, pred_mask):
    """Computes the intersection over union (IOU) between two masks."""
    intersection = (true_mask & pred_mask).sum()
    union = (true_mask | pred_mask).sum()
    return intersection / union

def compute_total_ious(true_mask, pred_mask, visibility, num_slots):
    true_mask_one_hot = F.one_hot(true_mask.long(), num_classes=visibility.size(1)).float()
    pred_mask_one_hot = F.one_hot(pred_mask.long(), num_classes=num_slots).float()

    true_mask_one_hot = true_mask_one_hot.permute(0,4,1,2,3)
    pred_mask_one_hot = pred_mask_one_hot.permute(0,4,1,2,3)

    visibility[:, 0, :] = 0
    indices = compute_matching(true_mask_one_hot, pred_mask_one_hot, visibility)
    miou_list = [] 
    mbo_list = []

    bs = true_mask.size(0)
    for b_idx in range(bs):
        miou_list_no_bg = []
        mbo_list_no_bg = []

        for gt_idx, pred_idx in zip(indices[b_idx][0], indices[b_idx][1]):
            if visibility[b_idx, ..., 0][gt_idx] == 0:  # no gt_idx - 1 here because we added the background to the beginning
                continue

            gt_map = (true_mask[b_idx] == gt_idx)
            pred_map = (pred_mask[b_idx] == pred_idx)

            miou_list_no_bg.append((gt_map & pred_map).sum() / (gt_map | pred_map).sum())

            iou_all = (gt_map[None] & pred_mask_one_hot[b_idx].bool()).sum((-2, -1)) / \
                        (gt_map[None] | pred_mask_one_hot[b_idx].bool()).sum((-2, -1))
            mbo_list_no_bg.append(iou_all.max())

        if len(miou_list_no_bg)!=0:
            miou_list.append(torch.tensor(miou_list_no_bg))
            mbo_list.append(torch.tensor(mbo_list_no_bg))

    return miou_list, mbo_list


import torch


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
        x: pytorch tensor
    Returns:
        x: pytorch tensor, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
        x: pytorch tensor, with shape [m, d]
        y: pytorch tensor, with shape [n, d]
    Returns:
        dist: pytorch tensor, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def compute_dist(array1, array2, dist_type='cosine', cos_to_normalize=True):
    """
    Args:
        array1: pytorch tensor, with shape [m, d]
        array2: pytorch tensor, with shape [n, d]
    Returns:
        dist: pytorch tensor, with shape [m, n]
    """
    if dist_type == 'cosine':
        if cos_to_normalize:
            array1 = normalize(array1, axis=1)
            array2 = normalize(array2, axis=1)
        dist = - torch.mm(array1, array2.t())
        # Turn distance into positive value
        dist += 1
    elif dist_type == 'euclidean':
        dist = euclidean_dist(array1, array2)
    else:
        raise NotImplementedError
    return dist


def compute_dist_with_qg_visibility(array1, array2, vis1, vis2, dist_type='cosine', avg_by_vis_num=True):
    """Compute the euclidean or cosine distance of all pairs, considering part visibility.
    In this version, the distance of a <query part, gallery part> pair if only calculated when
    both are visible. And finally, distance of a <query image, gallery image> pair is set to a
    large value, if they do not have commonly visible part.
    Args:
        array1: numpy array with shape [m1, p, d]
        array2: numpy array with shape [m2, p, d]
        vis1: numpy array with shape [m1, p], p is num_parts
        vis2: numpy array with shape [m2, p], p is num_parts
        dist_type: one of ['cosine', 'euclidean']
        avg_by_vis_num: for each <query_image, gallery_image> distance, average the
            summed distance by the number of commonly visible parts.
    Returns:
        dist: numpy array with shape [m1, m2]
    """
    err_msg = "array1.shape = {}, vis1.shape = {}, array2.shape = {}, vis2.shape = {}"\
        .format(array1.shape, vis1.shape, array2.shape, vis2.shape)
    assert len(array1.shape) == 3, err_msg
    assert len(array2.shape) == 3, err_msg
    assert array1.shape[0] == vis1.shape[0], err_msg
    assert array2.shape[0] == vis2.shape[0], err_msg
    assert array1.shape[2] == array2.shape[2], err_msg
    assert array1.shape[1] == array2.shape[1] == vis1.shape[1] == vis2.shape[1], err_msg
    m1 = array1.shape[0]
    m2 = array2.shape[0]
    p = vis1.shape[1]
    d = array1.shape[2]
    dist = 0
    vis_sum = 0
    for i in range(p):
        # [m1, m2]
        dist_ = compute_dist(array1[:, i, :], array2[:, i, :], dist_type=dist_type)
        q_visible = vis1[:, i].unsqueeze(1).repeat([1, m2]) != 0
        g_visible = vis2[:, i].unsqueeze(0).repeat([m1, 1]) != 0
        visible = (q_visible & g_visible).float()
        dist += dist_ * visible
        vis_sum += visible
    if avg_by_vis_num:
        dist /= (vis_sum + 1e-8)
        # If this <query image, gallery image> pair does not have common
        # visible part, set their distance to a very large value.
        # dist[vis_sum == 0] = 10000
    return dist, vis_sum
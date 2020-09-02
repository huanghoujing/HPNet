"""Numpy version of euclidean distance, etc.
Notice the input/output shape of methods, so that you can better understand
the meaning of these methods."""
import numpy as np


def normalize(nparray, order=2, axis=0):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


def compute_dist(array1, array2, dist_type='cosine', cos_to_normalize=True):
    """Compute the euclidean or cosine distance of all pairs.
    Args:
        array1: numpy array with shape [m1, n]
        array2: numpy array with shape [m2, n]
        dist_type: one of ['cosine', 'euclidean']
    Returns:
        dist: numpy array with shape [m1, m2]
    """
    if dist_type == 'cosine':
        if cos_to_normalize:
            array1 = normalize(array1, axis=1)
            array2 = normalize(array2, axis=1)
        dist = - np.matmul(array1, array2.T)
        # Turn distance into positive value
        dist += 1
    elif dist_type == 'euclidean':
        # shape [m1, 1]
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
        dist[dist < 0] = 0
        # Print('Debug why there is warning in np.sqrt')
        # np.seterr(all='raise')
        # for x in dist.flatten():
        #     try:
        #         np.sqrt(x)
        #     except:
        #         print(x)
        # Setting `out=dist` saves 1x memory size of `dist`
        np.sqrt(dist, out=dist)
    else:
        raise NotImplementedError
    return dist


def compute_dist_with_q_visibility(array1, array2, vis1, vis2, dist_type='cosine', avg_by_vis_num=True):
    """Compute the euclidean or cosine distance of all pairs, considering part visibility.
    In this version, if a query image does not has some part, don't calculate distance for this part.
    If a query has one part that gallery does not have, we can optionally set the part distance to some
    prior value, e.g. the mean distance of this part.
    Args:
        array1: numpy array with shape [m1, n]
        array2: numpy array with shape [m2, n]
        vis1: numpy array with shape [m1, p], p is num_parts
        vis2: numpy array with shape [m2, p], p is num_parts
        dist_type: one of ['cosine', 'euclidean']
        avg_by_vis_num: for each <query_image, gallery_image> distance, average the
            summed distance by the number of visible parts in query_image
    Returns:
        dist: numpy array with shape [m1, m2]
    """
    err_msg = "array1.shape = {}, vis1.shape = {}, array2.shape = {}, vis2.shape = {}"\
        .format(array1.shape, vis1.shape, array2.shape, vis2.shape)
    assert array1.shape[0] == vis1.shape[0], err_msg
    assert array2.shape[0] == vis2.shape[0], err_msg
    assert vis1.shape[1] == vis2.shape[1], err_msg
    assert array1.shape[1] % vis1.shape[1] == 0, err_msg
    assert array2.shape[1] % vis2.shape[1] == 0, err_msg
    m1 = array1.shape[0]
    m2 = array2.shape[0]
    p = vis1.shape[1]
    d = int(array1.shape[1] / vis1.shape[1])

    array1 = array1.reshape([m1, p, d])
    array2 = array2.reshape([m2, p, d])
    dist = 0
    for i in range(p):
        # [m1, m2]
        dist_ = compute_dist(array1[:, i, :], array2[:, i, :], dist_type=dist_type)
        q_invisible = vis1[:, i][:, np.newaxis].repeat(m2, 1) == 0
        dist_[q_invisible] = 0
        dist += dist_
    if avg_by_vis_num:
        dist /= (np.sum(vis1, axis=1, keepdims=True) + 1e-8)
    return dist


def compute_dist_with_qg_visibility(array1, array2, vis1, vis2, dist_type='cosine', avg_by_vis_num=True):
    """Compute the euclidean or cosine distance of all pairs, considering part visibility.
    In this version, the distance of a <query part, gallery part> pair if only calculated when
    both are visible. And finally, distance of a <query image, gallery image> pair is set to a
    large value, if they do not have commonly visible part.
    Args:
        array1: numpy array with shape [m1, n]
        array2: numpy array with shape [m2, n]
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
    assert array1.shape[0] == vis1.shape[0], err_msg
    assert array2.shape[0] == vis2.shape[0], err_msg
    assert vis1.shape[1] == vis2.shape[1], err_msg
    assert array1.shape[1] % vis1.shape[1] == 0, err_msg
    assert array2.shape[1] % vis2.shape[1] == 0, err_msg
    m1 = array1.shape[0]
    m2 = array2.shape[0]
    p = vis1.shape[1]
    d = int(array1.shape[1] / vis1.shape[1])

    array1 = array1.reshape([m1, p, d])
    array2 = array2.reshape([m2, p, d])
    dist = 0
    vis_sum = 0
    for i in range(p):
        # [m1, m2]
        dist_ = compute_dist(array1[:, i, :], array2[:, i, :], dist_type=dist_type)
        q_visible = vis1[:, i][:, np.newaxis].repeat(m2, 1) != 0
        g_visible = vis2[:, i][np.newaxis, :].repeat(m1, 0) != 0
        visible = (q_visible & g_visible).astype(np.float32)
        dist += dist_ * visible
        vis_sum += visible
    if avg_by_vis_num:
        dist /= (vis_sum + 1e-8)
        # If this <query image, gallery image> pair does not have common
        # visible part, set their distance to a very large value.
        dist[vis_sum == 0] = 10000
    return dist


def compute_dist_with_cfg(dic, cfg):
    if cfg.vis_type == 'None':
        return compute_dist(dic['q_feat'], dic['g_feat'])
    elif cfg.vis_type == 'qvis':
        return compute_dist_with_q_visibility(dic['q_feat'], dic['g_feat'], dic['q_visible'], dic['g_visible'])
    elif cfg.vis_type == 'qgvis':
        return compute_dist_with_qg_visibility(dic['q_feat'], dic['g_feat'], dic['q_visible'], dic['g_visible'])
    elif cfg.vis_type == 'qvis+qgvis':
        return compute_dist_with_q_visibility(dic['q_feat'], dic['g_feat'], dic['q_visible'], dic['g_visible']) + compute_dist_with_qg_visibility(dic['q_feat'], dic['g_feat'], dic['q_visible'], dic['g_visible'])
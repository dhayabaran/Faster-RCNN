import numpy as np

def loc2bbox(source_bbox, offset):
    if source_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=offset.dtype)
    source_bbox = source_bbox.astype(source_bbox.dtype, copy=False)

    source_height = source_bbox[:, 2] - source_bbox[:, 0]
    source_width = source_bbox[:, 3] - source_bbox[:, 1]
    source_ctr_y = source_bbox[:, 0] + 0.5 * source_height
    source_ctr_x = source_bbox[:, 1] + 0.5 * source_width
    
    dy = offset[:, 0::4]
    dx = offset[:, 1::4]
    dh = offset[:, 2::4]
    dw = offset[:, 3::4]
    
    ctr_y = dy * source_height[:, np.newaxis] + source_ctr_y[:, np.newaxis]
    ctr_x = dx * source_width[:, np.newaxis] + source_ctr_x[:, np.newaxis]
    h = np.exp(dh) * source_height[:, np.newaxis]
    w = np.exp(dw) * source_width[:, np.newaxis]
    dst_bbox = np.zeros(offset.shape, dtype = offset.dtype)
    
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w
    
    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()

    return loc

def bbox_iou(bbox1, bbox2):
    if bbox1.shape[1] != 4 or bbox2.shape[1] != 4:
        raise IndexError

        # top left
    tl = np.maximum(bbox1[:, None, :2], bbox2[:, :2])
    # bottom right
    br = np.minimum(bbox1[:, None, 2:], bbox2[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox1[:, 2:] - bbox1[:, :2], axis=1)
    area_b = np.prod(bbox2[:, 2:] - bbox2[:, :2], axis=1)


    return area_i / (area_a[:, None] + area_b - area_i)


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):

    py = base_size / 2.
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base
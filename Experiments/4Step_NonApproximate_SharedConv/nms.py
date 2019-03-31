import numpy as np
import cupy as cp

def non_maximum_suppression(roi, thresh=0.7, score=None):
    """Pure Python NMS baseline.
    roi: (N, 4)
    score: None or (N,)
    """
    #---------- debug
    assert isinstance(roi, np.ndarray)
    assert (score is None) or (isinstance(score, np.ndarray))
    assert len(roi.shape) == 2
    assert score is None or len(score.shape) == 1

    #----------
    x1 = roi[:, 0]
    y1 = roi[:, 1]
    x2 = roi[:, 2]
    y2 = roi[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    if score is None:    # roi are already sorted in large --> small order
        order = np.arange(roi.shape[0])
    else:               # roi are not sorted
        order = score.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]

    return np.asarray(keep).astype(int)
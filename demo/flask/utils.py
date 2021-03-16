import numpy as np

def get_type_confidence(result, class_name, score_thr):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    scores = bboxes[:, -1]

    type_confidence = []
    for i in range(len(labels)):
        if scores[i] > score_thr:
            type_confidence.append({'type': class_name[labels[i]], 'confidence': scores[i]})
    return type_confidence
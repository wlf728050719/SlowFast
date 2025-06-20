import numpy as np
from sklearn.metrics import average_precision_score

def compute_ap(gt_labels, pred_scores, num_classes):
    """
    计算每个类别的 AP (Average Precision)
    Args:
        gt_labels: 真实标签 (n_samples,)
        pred_scores: 预测概率 (n_samples, n_classes)
        num_classes: 类别数
    Returns:
        list: 每个类别的 AP
    """
    ap_list = []
    for cls in range(num_classes):
        y_true = (gt_labels == cls).astype(np.int32)
        y_score = pred_scores[:, cls]
        if np.sum(y_true) > 0:  # 确保正样本存在
            ap = average_precision_score(y_true, y_score)
        else:
            ap = 0.0
        ap_list.append(ap)
    return ap_list

def compute_map(gt_labels, pred_scores, num_classes):
    """
    计算 mAP (Mean Average Precision)
    """
    ap_list = compute_ap(gt_labels, pred_scores, num_classes)
    return np.mean(ap_list)
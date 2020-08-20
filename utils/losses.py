import torch
def stochastic_precision(pred, target):
    # print(pred)
    # print(target)
    total = 0#2
    hit = 0#1
    for y_pred, y in zip(pred, target):
        # y_pred = torch.ceil(y_pred) if y_pred > 0.5 else torch.floor(y_pred)

        if y_pred == 1.0:
            total += 1
            if y_pred == y:
                hit += 1
    print("pre", hit, total)
    return hit/total

def stochastic_recall(pred, target):
    total = 0
    hit = 0
    for y_pred, y in zip(pred, target):
        if y == 1.0:
            total += 1
            hit += y_pred
    print("recall", hit, total)
    return -hit/total


def f1_loss(y_pred, y_true):
    tp = torch.sum(y_true * y_pred, dim=0)
    tn = torch.sum((1 - y_true) * (1 - y_pred), dim=0)
    fp = torch.sum((1 - y_true) * y_pred, dim=0)
    fn = torch.sum(y_true * (1 - y_pred), dim=0)

    p = tp / (tp + fp + float(1e-7))
    r = tp / (tp + fn + float(1e-7))

    f1 = 2 * p * r / (p + r + float(1e-7))
    # f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    # return 1 - K.mean(f1)
    return 1 - torch.mean(f1)
def xr_loss(y_pred, y_true):
    tp = torch.sum(y_true * y_pred, dim=0)
    # tn = torch.sum((1 - y_true) * (1 - y_pred), dim=0)
    # fp = torch.sum((1 - y_true) * y_pred, dim=0)
    fn = torch.sum(y_true * (1 - y_pred), dim=0)

    # p = tp / (tp + fp + float(1e-7))
    r = tp / (tp + fn + float(1e-7))

    # f1 = 2 * p * r / (p + r + float(1e-7))
    # f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    # return 1 - K.mean(f1)
    return 1 - torch.mean(r)
# def f1_loss(input, target):


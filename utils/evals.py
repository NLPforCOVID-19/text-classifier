import torch
from torch.nn import functional
from tqdm import tqdm


# def eval_by_metric

def print_evals(results, targets, logger):
    output_tag = ["IsRelated", "Topic-感染状況", "Topic-予防・緊急事態宣言", "Topic-症状・治療・検査など医療情報", "Topic-経済・福祉政策", "Topic-休校・オンライン授業", " Topic-芸能・スポーツ",
                  "Topic-デマに関する記事", "Topic-その他", "Clarity", "Usefulness"]

    for i in range(results.size(1)):
        logger.debug("Accuracy for tag {}: {}".format(output_tag[i], compute_accuracy(results[:, i], targets[:, i])))
        # p, r = compute_precision(results[:, 0], targets[:, 0]), compute_recall(results[:, 0], targets[:, 0])
        f, p, r = prt_metric(results[:, i], targets[:, i])#, compute_recall(results[:, i], targets[:, i])
        logger.debug("P, R, F : {} {} {}".format(p, r, f))

    # logger.debug("Accuracy for tag IsRelated: {}".format(compute_accuracy(results[:, 0], targets[:,0])))
    # # p, r = compute_precision(results[:, 0], targets[:, 0]), compute_recall(results[:, 0], targets[:, 0])
    # f, p, r = prt_metric(results[:, 8], targets[:, 8]), compute_recall(results[:, 8], targets[:, 8])
    # logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))
    #
    # logger.debug("Accuracy for tag Topic-感染状況: {}".format(compute_accuracy(results[:, 1], targets[:, 1])))
    # # p, r = compute_precision(results[:, 1], targets[:, 1]), compute_recall(results[:, 1], targets[:, 1])
    # f, p, r = prt_metric(results[:, 8], targets[:, 8]), compute_recall(results[:, 8], targets[:, 8])
    # logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))
    #
    # logger.debug("Accuracy for tag Topic-予防・緊急事態宣言: {}".format(compute_accuracy(results[:, 2], targets[:, 2])))
    # # p, r = compute_precision(results[:, 2], targets[:, 2]), compute_recall(results[:, 2], targets[:, 2])
    # f, p, r = prt_metric(results[:, 8], targets[:, 8]), compute_recall(results[:, 8], targets[:, 8])
    # logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))
    # # logger.debug(torch.ceil(results[:3,2]), targets[:3, 2])
    #
    # logger.debug("Accuracy for tag Topic-症状・治療・検査など医療情報: {}".format(compute_accuracy(results[:, 3], targets[:, 3])))
    # # p, r = compute_precision(results[:, 3], targets[:, 3]), compute_recall(results[:, 3], targets[:, 3])
    # f, p, r = prt_metric(results[:, 8], targets[:, 8]), compute_recall(results[:, 8], targets[:, 8])
    # logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))
    #
    # logger.debug("Accuracy for tag Topic-経済・福祉政策: {}".format(compute_accuracy(results[:, 4], targets[:, 4])))
    # # p, r = compute_precision(results[:, 4], targets[:, 4]), compute_recall(results[:, 4], targets[:, 4])
    # f, p, r = prt_metric(results[:, 8], targets[:, 8]), compute_recall(results[:, 8], targets[:, 8])
    # logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))
    #
    # logger.debug("Accuracy for tag Topic-休校・オンライン授業: {}".format(compute_accuracy(results[:, 5], targets[:, 5])))
    # # p, r = compute_precision(results[:, 5], targets[:, 5]), compute_recall(results[:, 5], targets[:, 5])
    # f, p, r = prt_metric(results[:, 8], targets[:, 8]), compute_recall(results[:, 8], targets[:, 8])
    # logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))
    #
    # logger.debug("Accuracy for tag Topic-芸能・スポーツ: {}".format(compute_accuracy(results[:, 6], targets[:, 6])))
    # # p, r = compute_precision(results[:, 6], targets[:, 6]), compute_recall(results[:, 6], targets[:, 6])
    # f, p, r = prt_metric(results[:, 8], targets[:, 8]), compute_recall(results[:, 8], targets[:, 8])
    # logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))
    #
    #
    # logger.debug("Accuracy for tag Topic-デマに関する記事: {}".format(compute_accuracy(results[:, 7], targets[:, 7])))
    # # p, r = compute_precision(results[:, 7], targets[:, 7]), compute_recall(results[:, 7], targets[:, 7])
    # f, p, r = prt_metric(results[:, 8], targets[:, 8]), compute_recall(results[:, 8], targets[:, 8])
    # logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))
    #
    #
    # logger.debug("Accuracy for tag Topic-その他: {}".format(compute_accuracy(results[:, 8], targets[:, 8])))
    # # p, r = compute_precision(results[:, 8], targets[:, 8]), compute_recall(results[:, 8], targets[:, 8])
    # f, p, r = prt_metric(results[:, 8], targets[:, 8]), compute_recall(results[:, 8], targets[:, 8])
    # logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))
    #
    # logger.debug("Accuracy for tag Clarity: {}".format(compute_accuracy(results[:, 9], targets[:, 9])))
    # # p, r = compute_precision(results[:, 9], targets[:, 9]), compute_recall(results[:, 9], targets[:, 9])
    # f, p, r = prt_metric(results[:, 8], targets[:, 8]), compute_recall(results[:, 8], targets[:, 8])
    # logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))
    #
    # logger.debug("Accuracy for tag Usefulness: {}".format(compute_accuracy(results[:, 10], targets[:, 10])))
    # # p, r = compute_precision(results[:, 10], targets[:, 10]), compute_recall(results[:, 10], targets[:, 10])
    # f, p, r = prt_metric(results[:, 8], targets[:, 8]), compute_recall(results[:, 8], targets[:, 8])
    # logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))

def get_tags_from_dataset(dataset):
    tags = []

    for idx in range(len(dataset)):
        doc, tag = dataset[idx]
        isRelated, clarity, usefulness, topics = tag
        tags.append(torch.cat((isRelated, topics, clarity, usefulness), dim=0))
    return torch.stack(tags, dim=0)

def compute_precision(pred, target):
    # print(pred)
    # print(target)
    total = 2
    hit = 1
    for y_pred, y in zip(pred, target):
        y_pred = torch.ceil(y_pred) if y_pred > 0.5 else torch.floor(y_pred)
        if y_pred == 1.0:
            total += 1
            if y_pred == y:
                hit += 1
    print("pre", hit, total)
    return hit/total

def compute_recall(pred, target):
    total = 2
    hit = 1
    for y_pred, y in zip(pred, target):
        y_pred = torch.ceil(y_pred) if y_pred > 0.5 else torch.floor(y_pred)
        if y == 1.0:
            total += 1
            if y_pred == y:
                hit += 1
    print("recall", hit, total)
    return hit/total

def compute_accuracy(pred, target):
    correct = 0
    for y_pred, y in zip(pred, target):
        y_pred = torch.ceil(y_pred) if y_pred>0.5 else torch.floor(y_pred)
        if y_pred == y :
            correct+=1
    return correct/len(pred)


def prt_metric(y_pred, y_true):
    y_pred = torch.round(y_pred)
    print(y_true, y_pred)
    tp = torch.sum(y_true*y_pred, dim=0)
    tn = torch.sum((1-y_true)*(1-y_pred), dim=0)
    fp = torch.sum((1-y_true)*y_pred, dim=0)
    fn = torch.sum(y_true*(1-y_pred), dim=0)

    print("tp, tn, fp, fn: {} {} {} {}".format(tp, tn, fp, fn))

    p = tp / (tp + fp + float(1e-7))
    r = tp / (tp + fn + float(1e-7))

    f1 = 2*p*r / (p+r+float(1e-7))
    # f1 = torch.where(torch.is_nan(f1), torch.zeros_like(f1), f1)
    return torch.mean(f1), torch.mean(p), torch.mean(r)
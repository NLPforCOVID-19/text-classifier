import torch
from torch.nn import functional
from tqdm import tqdm


# def eval_by_metric

def print_evals(results, targets, logger):
    logger.debug("Accuracy for tag IsRelated: {}".format(compute_accuracy(results[:, 0], targets[:,0])))
    p, r = compute_precision(results[:, 0], targets[:, 0]), compute_recall(results[:, 0], targets[:, 0])
    logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))

    logger.debug("Accuracy for tag Topic-感染状況: {}".format(compute_accuracy(results[:, 1], targets[:, 1])))
    p, r = compute_precision(results[:, 1], targets[:, 1]), compute_recall(results[:, 1], targets[:, 1])
    logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))

    logger.debug("Accuracy for tag Topic-予防・緊急事態宣言: {}".format(compute_accuracy(results[:, 2], targets[:, 2])))
    p, r = compute_precision(results[:, 2], targets[:, 2]), compute_recall(results[:, 2], targets[:, 2])
    logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))
    # logger.debug(torch.ceil(results[:3,2]), targets[:3, 2])

    logger.debug("Accuracy for tag Topic-症状・治療・検査など医療情報: {}".format(compute_accuracy(results[:, 3], targets[:, 3])))
    p, r = compute_precision(results[:, 3], targets[:, 3]), compute_recall(results[:, 3], targets[:, 3])
    logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))

    logger.debug("Accuracy for tag Topic-経済・福祉政策: {}".format(compute_accuracy(results[:, 4], targets[:, 4])))
    p, r = compute_precision(results[:, 4], targets[:, 4]), compute_recall(results[:, 4], targets[:, 4])
    logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))

    logger.debug("Accuracy for tag Topic-休校・オンライン授業: {}".format(compute_accuracy(results[:, 5], targets[:, 5])))
    p, r = compute_precision(results[:, 5], targets[:, 5]), compute_recall(results[:, 5], targets[:, 5])
    logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))

    logger.debug("Accuracy for tag Topic-芸能・スポーツ: {}".format(compute_accuracy(results[:, 6], targets[:, 6])))
    p, r = compute_precision(results[:, 6], targets[:, 6]), compute_recall(results[:, 6], targets[:, 6])
    logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))

    logger.debug("Accuracy for tag Topic-デマに関する記事: {}".format(compute_accuracy(results[:, 7], targets[:, 7])))
    p, r = compute_precision(results[:, 7], targets[:, 7]), compute_recall(results[:, 7], targets[:, 7])
    logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))

    logger.debug("Accuracy for tag Topic-その他: {}".format(compute_accuracy(results[:, 8], targets[:, 8])))
    p, r = compute_precision(results[:, 8], targets[:, 8]), compute_recall(results[:, 8], targets[:, 8])
    logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))

    logger.debug("Accuracy for tag Clarity: {}".format(compute_accuracy(results[:, 9], targets[:, 9])))
    p, r = compute_precision(results[:, 9], targets[:, 9]), compute_recall(results[:, 9], targets[:, 9])
    logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))

    logger.debug("Accuracy for tag Usefulness: {}".format(compute_accuracy(results[:, 10], targets[:, 10])))
    p, r = compute_precision(results[:, 10], targets[:, 10]), compute_recall(results[:, 10], targets[:, 10])
    logger.debug("P, R, F : {} {} {}".format(p,r, 2*(p*r/(p+r))))

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

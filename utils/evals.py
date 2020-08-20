import torch
from torch.nn import functional
from tqdm import tqdm


# def eval_by_metric

def print_evals(results, targets, writer, details_best, n_iter):
    output_tag = ["is_about_covid_19", "is_useful", "is_clear", "is_about_false_rumor", "Topic-感染状況", "Topic-予防・緊急事態宣言", "Topic-症状・治療・検査など医療情報", "Topic-経済・福祉政策", "Topic-休校・オンライン授業", " Topic-芸能・スポーツ",
                  "Topic-デマに関する記事", "Topic-その他"]
    details = []
    # print(results.size(), targets.size())
    for i in range(results.size(1)):
        # logger.debug("Accuracy for tag {}: {}".format(output_tag[i], compute_accuracy(results[:, i], targets[:, i])))
        # p, r = compute_precision(results[:, 0], targets[:, 0]), compute_recall(results[:, 0], targets[:, 0])
        f, p, r = prt_metric(results[:, i], targets[:, i], writer, n_iter)#, compute_recall(results[:, i], targets[:, i])
        writer.add_scalar('Evaluation/F1_{}'.format(output_tag[i]), f, n_iter)
        writer.add_scalar('Evaluation/Precision_{}'.format(output_tag[i]), p, n_iter)
        writer.add_scalar('Evaluation/Recall_{}'.format(output_tag[i]), r, n_iter)
        writer.add_scalar('Evaluation/Recorded F1_{}'.format(output_tag[i]), details_best[i], n_iter)
        # logger.debug("P, R, F : {} {} {}, Best F {}:".format(p, r, f, details_best[i]))
        details.append(f)
    return details
def get_fs(results, targets, writer):
    output_tag = ["is_about_covid_19", "is_useful", "is_clear", "is_about_false_rumor", "Topic-感染状況", "Topic-予防・緊急事態宣言", "Topic-症状・治療・検査など医療情報", "Topic-経済・福祉政策", "Topic-休校・オンライン授業", " Topic-芸能・スポーツ",
                  "Topic-デマに関する記事", "Topic-その他"]
    f_all = 0.0
    # print(results.size(), targets.size())
    for i in range(results.size(1)):
        # logger.debug("Accuracy for tag {}: {}".format(output_tag[i], compute_accuracy(results[:, i], targets[:, i])))
        # p, r = compute_precision(results[:, 0], targets[:, 0]), compute_recall(results[:, 0], targets[:, 0])
        f, p, r = prt_metric(results[:, i], targets[:, i], writer, -1)#, compute_recall(results[:, i], targets[:, i])
        # details.append()
        f_all += f
    return f_all

def get_tags_from_dataset(dataset):
    tags = []
    for idx in range(len(dataset)):
        doc, tag, lang = dataset[idx]
        tags.append(tag)
    return torch.stack(tags, dim=0)

# def compute_precision(pred, target):
#     total = 2
#     hit = 1
#     for y_pred, y in zip(pred, target):
#         y_pred = torch.ceil(y_pred) if y_pred > 0.5 else torch.floor(y_pred)
#         if y_pred == 1.0:
#             total += 1
#             if y_pred == y:
#                 hit += 1
#     print("pre", hit, total)
#     return hit/total
#
# def compute_recall(pred, target):
#     total = 2
#     hit = 1
#     for y_pred, y in zip(pred, target):
#         y_pred = torch.ceil(y_pred) if y_pred > 0.5 else torch.floor(y_pred)
#         if y == 1.0:
#             total += 1
#             if y_pred == y:
#                 hit += 1
#     print("recall", hit, total)
#     return hit/total
#
def compute_accuracy(pred, target):
    correct = 0
    for y_pred, y in zip(pred, target):
        y_pred = torch.ceil(y_pred) if y_pred>0.5 else torch.floor(y_pred)
        if y_pred == y :
            correct+=1
    return correct/len(pred)


def prt_metric(y_pred, y_true, writer, n_iter, log=False):
    y_pred = torch.round(y_pred)
    print(y_true, y_pred)
    tp = torch.sum(y_true*y_pred, dim=0)
    tn = torch.sum((1-y_true)*(1-y_pred), dim=0)
    fp = torch.sum((1-y_true)*y_pred, dim=0)
    fn = torch.sum(y_true*(1-y_pred), dim=0)
    if log:
        writer.add_scalar('Evaluation/tp', tp, n_iter)
        writer.add_scalar('Evaluation/tn', tn, n_iter)
        writer.add_scalar('Evaluation/fp', fp, n_iter)
        writer.add_scalar('Evaluation/fn', fn, n_iter)
    # logger.debug("tp, tn, fp, fn: {} {} {} {}".format(tp, tn, fp, fn))

    p = tp / (tp + fp + float(1e-7))
    r = tp / (tp + fn + float(1e-7))

    f1 = 2*p*r / (p+r+float(1e-7))
    # f1 = torch.where(torch.is_nan(f1), torch.zeros_like(f1), f1)
    return torch.mean(f1), torch.mean(p), torch.mean(r)
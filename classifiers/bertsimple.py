"""Text classifier using BERT"""

import logging
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger(__name__)

def init_bert(bert_path):
    """Initialize BERT model"""
    bert = BertModel.from_pretrained(bert_path)
    tokenizer = BertTokenizer(bert_path + "/vocab.txt", do_lower_case=False, do_basic_tokenize=False)
    return bert, tokenizer

def run_bert(text_file, classes, bert, tokenizer, batch_size=128, device='cpu'):
    """Load text file and run BERT to obtain features"""
    logger.info("Load data and compute document embeddings: %s", text_file)
    features = []  # document embeddings (= a list of [CLS] embeddings)
    labels = []    # class labels
    with open(text_file, "r") as tf, torch.no_grad():
        for sent_id, line in enumerate(tf):
            #if sent_id >= 20: break
            # read text data (list of word lists)
            line = line.strip()
            page = json.loads(line)
            wordlists = page['wordlists']
            wordlists = [s for s in wordlists if len(s) > 0]  # remove empty sentences
            if len(wordlists) == 0: continue  # ignore documents with zero sentences
            # read class labels if available
            if 'classes' in page:
                labels.append(torch.tensor([min(1, page['classes'][c]) for c in classes], dtype=float))  # truncate values > 1
            # convert texts into word IDs
            idlists = tokenizer.batch_encode_plus(wordlists, add_special_tokens=True, pad_to_max_length=True)
            #print(wordlists)
            #print(idlists)
            # run BERT to compute embeddings
            cls_embeddings = []  # [CLS] embeddings
            for i in range(0, len(idlists["input_ids"]), batch_size):  # run BERT on minibatches
                batch = torch.tensor(idlists["input_ids"][i:i+batch_size]).to(device)
                mask = torch.tensor(idlists["attention_mask"][i:i+batch_size]).to(device)
                last_layer, _ = bert(batch, attention_mask=mask)
                #print(last_layer)
                #print(last_layer.shape)
                cls = last_layer[:, 0, :]  # [CLS] embeddings of minibatch
                cls_embeddings.append(cls.cpu())
            #cls_embedding /= len(idlists["input_ids"])  # compute mean of [CLS] embeddings
            features.append(torch.cat(cls_embeddings))
    # features = torch.stack(features)
    # labels = torch.stack(labels)
    logger.info("Data size: %s", len(features))
    return features, labels

class BertMeanPoolClassifier(nn.Module):
    """Document embedding = mean of sentence embeddings"""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        return

    def forward(self, doc_features, mask):
        num_sentences = mask.sum(axis=1).unsqueeze(dim=1)  # sentence lengths
        sum = doc_features.sum(axis=1)    # sum of sentence embeddings
        mean = sum / num_sentences        # mean of sentence embeddings
        output = self.linear(mean)
        return output

class BertMeanMaxPoolClassifier(nn.Module):
    """Document embedding = mean and max of sentence embeddings"""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim * 2, output_dim)
        return

    def forward(self, doc_features, mask):
        num_sentences = mask.sum(axis=1).unsqueeze(dim=1)  # sentence lengths
        sum = doc_features.sum(axis=1)    # sum of sentence embeddings
        mean = sum / num_sentences        # mean of sentence embeddings
        max = doc_features.max(dim=1).values  # max of sentence embeddings
        output = self.linear(torch.cat([mean, max], dim=1))
        return output

def init_classifier(input_dim, output_dim):
    """Create an instance of a classifier
    input: document embeddings, output: class probabilities"""
    #return nn.Linear(input_dim, output_dim)  # linear classifier for now...
    #return BertMeanPoolClassifier(input_dim, output_dim)
    return BertMeanMaxPoolClassifier(input_dim, output_dim)

def make_batch(features):
    """Create a batch of sentence embeddings; input = documents * sentences * sent_features"""
    # maximum length of documents
    max_len = max([sentences.shape[0] for sentences in features])
    batch = torch.stack([torch.cat([sentences, torch.zeros([max_len - sentences.shape[0], sentences.shape[1]])]) for sentences in features])
    mask = torch.stack([torch.tensor([i < sentences.shape[0] for i in range(max_len)]) for sentences in features])
    return batch, mask

def learning_curve_mode(args, classes):
    """Learning curve mode"""
    logger.info("Learning curve mode")

    # initialize BERT models
    logger.info("Loading BERT model: %s", args.bert_model)
    bert, tokenizer = init_bert(args.bert_model)
    bert.eval()
    bert.to(args.device)

    # run BERT on text file to compute document embeddings
    features, labels = run_bert(args.text_file, classes, bert, tokenizer, batch_size=args.batch, device=args.device)
    if len(features) != len(labels):
        raise ValueError("Train mode requires labeled text data")
    train_all_features, val_features, train_all_labels, val_labels = train_test_split(features, labels, test_size=0.1, shuffle=True, random_state=100)
    num_all_train = len(train_all_labels)
    num_all_labels = sum(train_all_labels)
    logger.info("Num train: %s", num_all_train)
    logger.info("Labels: %s", ' '.join([f"{label}:{int(num)}" for label, num in zip(classes, num_all_labels)]))
    
    # train model varying data size
    dataset_indices = torch.randperm(num_all_train)
    for dataset_ratio in args.dataset_ratio:
        logger.info("Start training: %s", dataset_ratio)
        num_train = int(num_all_train * dataset_ratio)
        train_features = [train_all_features[i] for i in dataset_indices[:num_train]]
        train_labels = [train_all_labels[i] for i in dataset_indices[:num_train]]
        num_labels = sum(train_labels)
        classifier = init_classifier(train_features[0].shape[1], len(classes))
        classifier.to(args.device)
        if args.pos_weight:
            # weighting positive instances to deal with imbalanced labels (weight = # negative / # positive)
            pos_weight = torch.tensor([float(num_train - pos) / pos for pos in num_labels]).to(args.device)
            #print(pos_weight)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            # no pos_weight
            criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adagrad(classifier.parameters())
        # start training
        for epoch in range(args.epoch):
            total_loss = 0.0
            permutation = torch.randperm(num_train)
            for i in range(0, num_train, args.batch):  # run classifier on minibatch
                indices = permutation[i:i+args.batch]
                batch, mask = make_batch([train_features[i] for i in indices])
                batch = batch.to(args.device)
                mask = mask.to(args.device)
                gold = torch.stack([train_labels[i] for i in indices]).to(args.device)
                optimizer.zero_grad()
                output = classifier(batch, mask)   # class scores
                #print(output)
                #print(gold)
                loss = criterion(output, gold)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            total_loss /= (num_train // args.batch + 1)
            logger.info(f"Epoch {epoch+1}: loss={total_loss}")
            # compute validation loss and accuracy
            with torch.no_grad():
                val_loss = 0
                val_probs = []
                for i in range(0, len(val_features), args.batch):
                    val_batch, val_mask = make_batch(val_features[i:i+args.batch])
                    val_batch = val_batch.to(args.device)
                    val_mask = val_mask.to(args.device)
                    val_gold = torch.stack(val_labels[i:i+args.batch]).to(args.device)
                    val_output = classifier(val_batch, val_mask)
                    val_loss += float(criterion(val_output, val_gold).item())
                    val_probs.append(torch.sigmoid(val_output).cpu())
                val_loss /= (len(val_features) / args.batch + 1)
            prec, rec, fscore, _ = precision_recall_fscore_support(torch.stack(val_labels), (torch.cat(val_probs) > 0.5).to(dtype=float), zero_division=0)
            acc = ' '.join([f"{cl}={p:.3f}/{r:.3f}/{f:.3f}" for cl, p, r, f in zip(classes, prec, rec, fscore)])
            logger.info(f"val_loss={val_loss} acc: {acc}")
        logger.info("done")

        # output final accuracy
        logger.info("t=%s: %s", dataset_ratio, ' '.join([f"{cl}={p:.3f}/{r:.3f}/{f:.3f}" for cl, p, r, f in zip(classes, prec, rec, fscore)]))
    return

def train_mode(args, classes):
    """Train mode"""
    logger.info("Train mode")

    # initialize BERT models
    logger.info("Loading BERT model: %s", args.bert_model)
    bert, tokenizer = init_bert(args.bert_model)
    bert.eval()
    bert.to(args.device)

    # run BERT on text file to compute document embeddings
    features, labels = run_bert(args.text_file, classes, bert, tokenizer, batch_size=args.batch, device=args.device)
    if len(features) != len(labels):
        raise ValueError("Train mode requires labeled text data")
    train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.1, shuffle=True, random_state=100)
    num_train = len(train_labels)
    num_labels = sum(train_labels)
    logger.info("Num train: %s", num_train)
    logger.info("Labels: %s", ' '.join([f"{label}:{int(num)}" for label, num in zip(classes, num_labels)]))
    
    # train model
    logger.info("Start training")
    classifier = init_classifier(train_features[0].shape[1], len(classes))
    classifier.to(args.device)
    if args.pos_weight:
        # weighting positive instances to deal with imbalanced labels (weight = # negative / # positive)
        pos_weight = torch.tensor([float(num_train - pos) / pos for pos in num_labels]).to(args.device)
        #print(pos_weight)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        # no pos_weight
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adagrad(classifier.parameters())
    # start training
    for epoch in range(args.epoch):
        total_loss = 0.0
        permutation = torch.randperm(num_train)
        for i in range(0, num_train, args.batch):  # run classifier on minibatch
            indices = permutation[i:i+args.batch]
            batch, mask = make_batch([train_features[i] for i in indices])
            batch = batch.to(args.device)
            mask = mask.to(args.device)
            gold = torch.stack([train_labels[i] for i in indices]).to(args.device)
            optimizer.zero_grad()
            output = classifier(batch, mask)   # class scores
            #print(output)
            #print(gold)
            loss = criterion(output, gold)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        total_loss /= (num_train // args.batch + 1)
        logger.info(f"Epoch {epoch+1}: loss={total_loss}")
        # compute validation loss and accuracy
        with torch.no_grad():
            val_loss = 0
            val_probs = []
            for i in range(0, len(val_features), args.batch):
                val_batch, val_mask = make_batch(val_features[i:i+args.batch])
                val_batch = val_batch.to(args.device)
                val_mask = val_mask.to(args.device)
                val_gold = torch.stack(val_labels[i:i+args.batch]).to(args.device)
                val_output = classifier(val_batch, val_mask)
                val_loss += float(criterion(val_output, val_gold).item())
                val_probs.append(torch.sigmoid(val_output).cpu())
            val_loss /= (len(val_features) / args.batch + 1)
        prec, rec, fscore, _ = precision_recall_fscore_support(torch.stack(val_labels), (torch.cat(val_probs) > 0.5).to(dtype=float), zero_division=0)
        acc = ' '.join([f"{cl}={p:.3f}/{r:.3f}/{f:.3f}" for cl, p, r, f in zip(classes, prec, rec, fscore)])
        logger.info(f"val_loss={val_loss} acc: {acc}")
    logger.info("done")

    # save model
    logger.info("Save model: %s", args.model_path)
    torch.save(classifier.state_dict(), args.model_path)
    logger.info("done")
    return

def test_mode(args, classes):
    """Test mode"""
    logger.info("Test mode")
    
    # initialize BERT models
    logger.info("Loading BERT model: %s", args.bert_model)
    bert, tokenizer = init_bert(args.bert_model)
    bert.eval()
    bert.to(args.device)

    # run BERT on text file to compute document embeddings
    features, _ = run_bert(args.text_file, classes, bert, tokenizer, batch_size=args.batch, device=args.device)
    num_test = len(features)

    # run classifier
    logger.info("Start running")
    classifier = init_classifier(features[0].shape[1], len(classes))
    classifier.load_state_dict(torch.load(args.model_path))
    classifier.eval()
    classifier.to(args.device)
    outputs = []  # list of class probabilities
    with torch.no_grad():
        for i in range(0, len(features), args.batch):  # run classifier on minibatch
            batch, mask = make_batch(features[i:i+args.batch])
            batch = batch.to(args.device)
            mask = mask.to(args.device)
            scores = classifier(batch, mask)   # class scores
            probs = torch.sigmoid(scores)      # class probabilities
            outputs.extend(probs.cpu())

    # output class probabilities
    logger.info("Output: %s", args.output_file)
    with open(args.output_file, "w") as of:
        for output in outputs:
            result = {"classes": {label: val for label, val in zip(classes, output.tolist())}}
            of.write(json.dumps(result, ensure_ascii=False))
            of.write('\n')
    logger.info("done")
    return

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s: %(message)s')
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", choices=["train", "test", "learning"], help="Run mode")
    argparser.add_argument("--bert-model", required=True, help="Path to BERT model")
    argparser.add_argument("--model-path", required=True, help="Path to the classifier mdoel")
    argparser.add_argument("--text-file", required=True, help="Text file (JSONL)")
    argparser.add_argument("--classes-file", required=True, help="File of target classes (a class label per line)")
    argparser.add_argument("--output-file", help="Output file (JSONL); test mode only")
    argparser.add_argument("--gpu", action="store_true", help="Use GPU")
    argparser.add_argument("--batch", type=int, default=128, help="Batch size")
    argparser.add_argument("--epoch", type=int, default=30, help="Number of epochs")
    argparser.add_argument("--pos-weight", action="store_true", help="Use pos_weight in loss function")
    argparser.add_argument("--dataset-ratio", type=lambda x: list(map(float, x.split(','))), help="Ratio of data used for training (comma-separated float values; only valid for learning curve mode)")
    args = argparser.parse_args()
    if args.gpu and torch.cuda.is_available():
        logger.info("Using GPU")
        args.device = "cuda"
    else:
        logger.info("Using CPU")
        args.device = "cpu"

    # read classes file
    with open(args.classes_file, "r") as f:
        classes = [line.strip() for line in f]

    # start train or test
    if args.mode == "train":
        train_mode(args, classes)
    elif args.mode == "learning":
        if args.dataset_ratio is None:
            logger.error("Learning curve mode requires --dataset-ratio")
            sys.exit(1)
        learning_curve_mode(args, classes)
    else:
        assert args.mode == "test"
        if args.output_file is None:
            logger.error("Test mode requires --output-file")
            sys.exit(1)
        test_mode(args, classes)

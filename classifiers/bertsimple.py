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
    features = []  # document embeddings (= mean of [CLS] embeddings)
    labels = []    # class labels
    with open(text_file, "r") as tf, torch.no_grad():
        for line in tf:
            # read text data (list of word lists)
            line = line.strip()
            page = json.loads(line)
            wordlists = page['wordlists']
            # read class labels if available
            if 'classes' in page:
                labels.append(torch.tensor([min(1, page['classes'][c]) for c in classes], dtype=float))  # truncate values > 1
            # convert texts into word IDs
            idlists = tokenizer.batch_encode_plus(wordlists, add_special_tokens=True, pad_to_max_length=True)
            #print(wordlists)
            #print(idlists)
            # run BERT to compute embeddings
            cls_embedding = None  # sum of [CLS] embeddings
            for i in range(0, len(idlists["input_ids"]), batch_size):  # run BERT on minibatches
                batch = torch.tensor(idlists["input_ids"][i:i+batch_size]).to(device)
                mask = torch.tensor(idlists["attention_mask"][i:i+batch_size]).to(device)
                last_layer, _ = bert(batch, attention_mask=mask)
                #print(last_layer)
                #print(last_layer.shape)
                cls = last_layer[:, 0, :].sum(axis=0)  # sum of [CLS] embeddings of minibatch
                if cls_embedding is None:
                    cls_embedding = cls
                else:
                    cls_embedding += cls
            cls_embedding /= len(idlists["input_ids"])  # compute mean of [CLS] embeddings
            features.append(cls_embedding)
    features = torch.stack(features)
    labels = torch.stack(labels)
    logger.info("Data size: %s", features.shape[0])
    return features, labels

def init_classifier(input_dim, output_dim):
    """Create an instance of a classifier
    input: document embeddings, output: class probabilities"""
    return nn.Linear(input_dim, output_dim)  # linear classifier for now...

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
    num_train = train_labels.shape[0]
    num_labels = train_labels.sum(axis=0)
    logger.info("Num train: %s", num_train)
    logger.info("Labels: %s", ' '.join([f"{label}:{num}" for label, num in zip(classes, num_labels)]))
    
    # train model
    logger.info("Start training")
    classifier = init_classifier(train_features[0].shape[0], len(classes))
    classifier.to(args.device)
    # weighting positive instances to deal with imbalanced labels (weight = # negative / # positive)
    pos_weight = torch.tensor([float(num_train - pos) / pos for pos in num_labels]).to(args.device)
    #print(pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adagrad(classifier.parameters())
    # start training
    for epoch in range(args.epoch):
        total_loss = 0.0
        for i in range(0, len(train_features), args.batch):  # run classifier on minibatch
            batch = train_features[i:i+args.batch].to(args.device)
            gold = train_labels[i:i+args.batch].to(args.device)
            optimizer.zero_grad()
            output = classifier(batch)   # class scores
            #print(output)
            #print(gold)
            loss = criterion(output, gold)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        logger.info(f"Epoch {epoch+1}: loss={total_loss}")
        # compute validation loss and accuracy
        val_features = val_features.to(args.device)
        val_labels = val_labels.to(args.device)
        val_output = classifier(val_features)
        val_loss = criterion(val_output, val_labels)
        val_probs = torch.sigmoid(val_output)
        prec, rec, fscore, _ = precision_recall_fscore_support(val_labels.cpu(), (val_probs.cpu() > 0.5).to(dtype=float), zero_division=0)
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
    num_test = features.shape[0]

    # run classifier
    logger.info("Start running")
    classifier = init_classifier(features[0].shape[0], len(classes))
    classifier.load_state_dict(torch.load(args.model_path))
    classifier.eval()
    classifier.to(args.device)
    outputs = []  # list of class probabilities
    with torch.no_grad():
        for i in range(0, len(features), args.batch):  # run classifier on minibatch
            batch = features[i:i+args.batch].to(args.device)
            scores = classifier(batch)    # class scores
            probs = torch.sigmoid(scores) # class probabilities
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
    argparser.add_argument("mode", choices=["train", "test"], help="Run mode")
    argparser.add_argument("--bert-model", required=True, help="Path to BERT model")
    argparser.add_argument("--model-path", required=True, help="Path to the classifier mdoel")
    argparser.add_argument("--text-file", required=True, help="Text file (JSONL)")
    argparser.add_argument("--classes-file", required=True, help="File of target classes (a class label per line)")
    argparser.add_argument("--output-file", help="Output file (JSONL); test mode only")
    argparser.add_argument("--gpu", action="store_true", help="Use GPU")
    argparser.add_argument("--batch", type=int, default=128, help="Batch size")
    argparser.add_argument("--epoch", type=int, default=30, help="Number of epochs")
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
    else:
        assert args.mode == "test"
        if args.output_file is None:
            logger.error("Test mode requires --output-file")
            sys.exit(1)
        test_mode(args, classes)

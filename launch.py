import os
import sys
import json
import glob
import pathlib
import argparse
import bs4
import datetime
import time
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from xml.etree import ElementTree
from pyknp import Juman

from preprocess.metadata import extract_meta
from preprocess.extracttext import extract_meta_add_text
from annotation_cleanup.classifier import extract_meta_add_keyword_classify
from classifiers.bertsimple import extract_meta_add_bert_info, init_bert, init_classifier


gpu_num = '3'
batch = 32

data_dir = '/mnt/hinoki/share/covid19'
input_dir = '/mnt/hinoki/share/covid19/run/new-translated-files'
output_dir = '/mnt/hinoki/share/covid19/topics_bert'
blacklist = '{}/url_black_list.txt'.format(data_dir)
sourceinfo_file = '{}/source_info_en.tsv'.format(data_dir)
output = '{}/output.jsonl'.format(data_dir)

classes_file = './classifiers/classes.txt'
bert_dir = './model'
bert_model = '{}/Japanese_L-12_H-768_A-12_E-30_BPE_transformers'.format(bert_dir) # dir for the bert model itself
bertsimple_model = '{}/classifier.pth'.format(bert_dir) # model of the classifier, bert+linear layers

keyword_file = './classifiers/keywords.txt'

translated_log_dir = '/mnt/hinoki/share/covid19/run/trans_log_song' 
ja_translated_list_file = '{}/trans_list.txt'.format(translated_log_dir)
en_translated_list_file = '{}/en_trans_list.txt'.format(translated_log_dir)

xml_log_dir = '/mnt/hinoki/share/covid19/run/new-xml-files'

output_dir = '/mnt/hinoki/share/covid19/run/topic_classification_log'
output_jsonl = '{}/output.jsonl'.format(output_dir)
output_txt = '{}/topic_classified_list.txt'.format(output_dir)
#output_errored_txt = '{}/topic_classified_list_errored.txt'.format(output_dir)

# 1. get process files from new_translate (Japanese)
# 2. for each file, generate metadata, generate bert classification result

def full_path_to_related_path(full_path):
    return full_path.replace('/mnt/hinoki/share/covid19','.')

def process_one_file(input_path):

    status = 0
    try:
        related_path = full_path_to_related_path(input_path)
        meta = extract_meta(data_dir, sourceinfo_file, related_path) 
        meta = extract_meta_add_text(data_dir, meta, juman)
        meta = extract_meta_add_keyword_classify(keyword_file, meta) 
        meta = extract_meta_add_bert_info(meta, classes, bert, classifier, tokenizer, batch, device)
    except:
        status = 1

    if (status == 0):
        with open(output_txt, "a+") as f:
            output_res = "{} {}\n".format(input_path.strip(), status)
            f.write(output_res)
        with open(output_jsonl, "a+") as f:
            output_res = json.dumps(meta, ensure_ascii=False)
            f.write(output_res.strip() + '\n')
    else:
        print ("error")
        with open(output_txt, "a+") as f:
            output_res = "{} {}\n".format(input_path.strip(), status)
            f.write(output_res)
        

def get_xml_converted_list(xml_log_dir):
    names = glob.glob("{}/new-xml-files*txt".format(xml_log_dir))
    all_xml_file = "{}/xmled_files.txt".format(xml_log_dir)
    xml_list = []
    with open(all_xml_file, "w") as of:
        for name in names:
            lines = open(name, "r").readlines()
            for line in lines:
                xml_list.append(line.strip())
                of.write(line.strip()+'\n')
    return xml_list

def get_feature(full_name):
    related_name = full_path_to_related_path(full_name)
    _, country, _, domain, *url_parts = pathlib.Path(related_name.strip()).parts
    suf = url_parts[-1].split('.')[-1]
    url_parts[-1] = url_parts[-1].replace(suf, '')
    feature = '/'.join(url_parts)
    return feature

def get_unprocessed_list():
    #_, country, _, domain, *url_parts = pathlib.Path(line.strip()).parts
    ja_translated_list = [line.strip().split()[0] for line in open(ja_translated_list_file, "r").readlines() if (line.strip().split()[2]=='0')]
    en_translated_list = [line.strip().split()[0] for line in open(en_translated_list_file, "r").readlines() if (line.strip().split()[2]=='0')]
    xml_converted_list = get_xml_converted_list(xml_log_dir)
    classified_list = [line.strip().split()[0] for line in open(output_txt, "r").readlines()]

    en_translated_dict = {}
    xml_converted_dict = {}
    classified_dict = {}
    unprocessed_list = []

    for en_translated_file in en_translated_list:
        feature = get_feature(en_translated_file)
        en_translated_dict[feature]=1
	
    for xml_file in xml_converted_list:
        feature = get_feature(xml_file)
        xml_converted_dict[feature]=1
	
    for classified_file in classified_list:
        feature = get_feature(classified_file)
        classified_dict[feature]=1

    for ja_translated_file in ja_translated_list:
        feature = get_feature(ja_translated_file)
        if (en_translated_dict.get(feature, 0) == 1) and \
        (xml_converted_dict.get(feature, 0) == 1) and \
        (classified_dict.get(feature, 0) == 0):
                unprocessed_list.append(ja_translated_file)

    unprocessed_list.reverse()
    return unprocessed_list


with open(classes_file, "r") as f:
    classes = [line.strip() for line in f]

device = "cuda:{}".format(gpu_num)
bert, tokenizer = init_bert(bert_model)
bert.eval()
bert.to(device)
classifier = init_classifier(768, 11)
classifier.load_state_dict(torch.load(bertsimple_model, map_location=device))
classifier.eval()
classifier.to(device)
juman = Juman()

itr = 0
while (1):
    itr += 1
    print ("Iteration {}".format(itr))
    unprocessed_list = get_unprocessed_list()
    print (len(unprocessed_list))
    for i, input_path in enumerate(unprocessed_list):
        print (i, input_path)
        print()
        process_one_file(input_path)
    time.sleep(1000)


test_flag = False 
if (test_flag == True):
    input_path = './html/fr/en_translated/www.lemonde.fr/signataires/francois-beguin/?page=3/2020/11/03-08-45/?page=3.html'
    related_path = full_path_to_related_path(input_path)
    meta = extract_meta(data_dir, sourceinfo_file, related_path) 
    meta = extract_meta_add_text(data_dir, meta, juman)
    meta = extract_meta_add_keyword_classify(keyword_file, meta) 
    meta = extract_meta_add_bert_info(meta, classes, bert, classifier, tokenizer, batch, device)
    print (meta)



import os
import csv
import codecs
import urllib.request
import numpy as np
        
# Some metadata about ASAP-AES dataset, in order to extract valid items.
set_valid = [str(i) for i in range(1, 9)]
score_valid = [str(i) for i in range(0, 61)]

# NER Tags for GloVe Implimentation
ner = ["@PERSON", "@ORGANIZATION", "@LOCATION", "@DATE",
       "@TIME", "@MONEY", "@PERCENT", "@MONTH", "@EMAIL",
       "@NUM", "@CAPS", "@DR", "@CITY", "@STATE"]

def load_glove(es_hp, path=None):  
    with codecs.open(path, 'r', 'UTF-8') as glove_file:
        glove_vectors = {}
        # Add numbers and NER embedding entries.
        for i in ner:
            glove_vectors[i] = np.random.randn(es_hp.w_dim)
        for item in glove_file.readlines():
            item_lst = item.strip().split(' ')
            word = item_lst[0]
            vec = [float(i) for i in item_lst[1:es_hp.w_dim+2]]
            glove_vectors[word] = np.array(vec)
    return glove_vectors

def load_asap(path="data/training_set.tsv", domain_id=None):
    if domain_id:
        print("[Loading] ASAP-AES domain {} dataset...".format(domain_id))
    else:
        print("Give a domain ID.")
    with codecs.open(path, "r", "ISO-8859-2") as asap_file:
        asap_reader = csv.DictReader(asap_file, delimiter="\t")
        # Extract valid items in the dataset.
        if not domain_id:
            asap_data = [item for item in asap_reader
                if item["essay"]
                and item["essay_set"] in set_valid
                and item["domain1_score"] in score_valid]
        else:
            asap_data = [item for item in asap_reader
                if item["essay"]
                and item["essay_set"] == str(domain_id)
                and item["domain1_score"] in score_valid]
    return asap_data
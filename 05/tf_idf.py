# coding: utf-8
import math
import pandas as pd


def compute_tf(word_dict, bow):
    tf_dict = {}
    bow_count = len(bow)
    for word, count in word_dict.items():
        tf_dict[word] = count/float(bow_count)
    return tf_dict


def compute_idf(doc_list):
    idf_dict = {}
    n = len(doc_list)
    idf_dict = dict.fromkeys(doc_list[0].keys(), 0)
    for doc in doc_list:
        for word, val in doc.items():
            if val > 0:
                idf_dict[word] += 1
    for word, val, in idf_dict.items():
        idf_dict[word] = math.log10(n/float(val))
    return idf_dict


def compute_tf_idf(tf_bow, idfs):
    tf_idf = {}
    for word, val in tf_bow.items():
        tf_idf[word] = val * idfs[word]
    return tf_idf


def load_tab(word_dict_a, word_dict_b):
    zoulou = pd.DataFrame([word_dict_a, word_dict_b])
    print(zoulou)


def main():
    docA = "The cat sat on my face"
    docB = "The dog sat on my bed"
    bow_a = docA.split(" ")
    bow_b = docB.split(" ")
    wordSet = set(bow_a).union(set(bow_b))
    word_dict_a = dict.fromkeys(wordSet, 0) 
    word_dict_b = dict.fromkeys(wordSet, 0)
    for word in bow_a:
        word_dict_a[word]+=1
    for word in bow_b:
        word_dict_b[word]+=1
    tfBow_a = compute_tf(word_dict_a, bow_a)
    tfBow_b = compute_tf(word_dict_b, bow_b)
    idfs = compute_idf([word_dict_a, word_dict_b])
    tfidfBow_a = compute_tf_idf(tfBow_a, idfs)
    tfidfBow_b = compute_tf_idf(tfBow_b, idfs)
    load_tab(tfidfBow_a, tfidfBow_b)
    sorted_tfidfBow_a = sorted(tfidfBow_a.items(), key=lambda tfidfBow_a: tfidfBow_a[1], reverse =True)
    sorted_tfidfBow_b = sorted(tfidfBow_b.items(), key=lambda tfidfBow_b: tfidfBow_b[1], reverse =True)

if __name__ == "__main__":
    main()
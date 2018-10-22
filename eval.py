from table import *


def evaluate(summarizer, doc, sum_name, n, *args, **kwargs):
    """
    doc          - document opened
    table        - table with tf-ids
    n            - summary len
    sum_name     - summary name (ex: Ext-ce94ja21-e.txt)
    """
    summry = summarizer(doc, n, *args, **kwargs)

    with open("data/sumarios_manuais/" + sum_name, 'r', encoding='latin1') as f:
        collection_sum = tokenize_sentences(f.read())
        dic = dict(tp=0, r=0, map=0, len_collection_sum=len(collection_sum) - 1)
        for i in range(len(summry)):
            if summry[i] in collection_sum:
                dic['tp'] += 1
                dic['map'] += pprecision(dic['tp'], i+1)
            dic['r'] += 1
        dic['map'] = dic['map']/(dic['len_collection_sum'])
        return dic


def evaluate_print(name, p, r, f1, map_):
    """
    name - string with the name of the method
    info - array: first position has an array with the precision for each doc
                  second position has an array with the recall for each doc
    dic  - dictionary has the document len and map
    """

    print(name + " average precision      = " + str(p))
    print(name + " average recall         = " + str(r))
    print(name + " f1 score               = " + str(f1))
    print(name + " mean average precision = " + str(map_))


def full_evaluate(summarizer, n, *args, **kwargs):
    P = []
    R = []
    AP = []

    path = 'data/flat_text/'
    for file in os.listdir(path):
        print(file)
        with open(path+file, 'r', encoding='latin1') as f:
            to_eval = f.read()
            result = evaluate(summarizer, to_eval, "Ext-"+file[3:], n, *args, **kwargs)

            P.append(precision(result['tp'], result['r'] - result['tp']))
            R.append(recall(result['tp'], result['len_collection_sum'] - result['tp']))
            AP.append(result['map'])

    avg_prec = sum(P) / float(len(P))
    avg_recall = sum(R) / float(len(R))
    map_ = sum(AP) / float(len(AP))
    f1_score = F1_score(avg_prec, avg_recall)

    return avg_prec, avg_recall, f1_score, map_

def evall(summry, sum_name):
    with open("data/sumarios_manuais/" + sum_name, 'r', encoding='latin1') as f:
        collection_sum = tokenize_sentences(f.read())
        for k in range(len(collection_sum)):
            collection_sum[k] = collection_sum[k].lower()
        dic = dict(tp=0, r=0, map=0, len_collection_sum=len(collection_sum))
        for i in range(len(summry)):
            if summry[i] in collection_sum:
                dic['tp'] += 1
                dic['map'] += pprecision(dic['tp'], i+1)
            dic['r'] += 1
        dic['map'] = dic['map']/dic['len_collection_sum']
        return dic

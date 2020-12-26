# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

import math
import numpy as np
import pandas as pd
# import pickle
import re
import time
# import xml.etree.ElementTree as ET

# from collections import Counter

data = 'data 400.csv'
train_opini = 'train_opini_filter.csv'

def readData(data):
    dframe = pd.read_csv(data, encoding = "cp1252")
    return dframe

data = readData(data)
dframe_train = data[:300]

def getListUlasan(col_ulasan):
    list_ulasan = col_ulasan['Ulasan'].to_list()
    return list_ulasan

list_ulasan_train = getListUlasan(dframe_train)

def preprocessing(ulasan):
    folded = [x.lower() for x in ulasan]
    return folded

ulasan_prepos_train = preprocessing(list_ulasan_train)

def decontracted(ulasan):
    # specific
    result = [re.sub(r"won't", "will not", ul) for ul in ulasan]
    result = [re.sub(r"can\'t", "can not", res) for res in result]

    # general
    result = [re.sub(r"n\'t", " not", res) for res in result]
    result = [re.sub(r"\'re", " are", res) for res in result]
    result = [re.sub(r"\'s", " is", res) for res in result]
    result = [re.sub(r"\'d", " would", res) for res in result]
    result = [re.sub(r"\'ll", " will", res) for res in result]
    result = [re.sub(r"\'t", " not", res) for res in result]
    result = [re.sub(r"\'ve", " have", res) for res in result]
    result = [re.sub(r"\'m", " am", res) for res in result]
    return result

ulasan_decontracted_train = decontracted(ulasan_prepos_train)

def postag(ulasan):
    tagged_reviews = []
    for each_review_text in ulasan:
        ulasan = nltk.word_tokenize(each_review_text)
        tagged_reviews.append(nltk.pos_tag(ulasan))
    return tagged_reviews

def postagUji(ulasan):
    for word in ulasan:
        ulasan = nltk.word_tokenize(word)
        tagged_reviews = nltk.pos_tag(ulasan)
    return tagged_reviews

ulasan_postag_train = postag(ulasan_decontracted_train)

def opini_rule(result_postag):
    results_tree = []
    grammar = "NP: {<DT|PP|CD|RB>?<JJ|JJR|JJS>*<NN|NNS|PRP|NNP|VB|IN|PRP\$>+<VBD|VBZ|VBN|VBP|VB|IN>*<JJ|JJS|RB>*<PRP|NN|NNS>*}"
    cp = nltk.RegexpParser(grammar)
    for tag in result_postag:
        results_tree.append(cp.parse(tag))
    return results_tree

def opini_rule_uji(result_postag):
    grammar = "NP: {<DT|PP|CD|RB>?<JJ|JJR|JJS>*<NN|NNS|PRP|NNP|VB|IN|PRP\$>+<VBD|VBZ|VBN|VBP|VB|IN>*<JJ|JJS|RB>*<PRP|NN|NNS>*}"
    cp = nltk.RegexpParser(grammar)
    results_tree = (cp.parse(result_postag))
    return results_tree

ulasan_tree_train = opini_rule(ulasan_postag_train)

def opini_extractor(result_rule):
    finish = []
    for result in range(len(result_rule)):
        temp = []
        finish.append([])
        for res in range(len(result_rule[result])):
            temp.append([])
            if type(result_rule[result][res]) == nltk.tree.Tree:
                for restu in result_rule[result][res]:
                    temp[res].append(restu[0])
            # print(temp)
            if len(temp[res]) >= 2:
                finish[result].append(" ".join(temp[res]))
    return finish

def opini_extractor_uji(result_rule):
    finish = []
    for res in result_rule:
        temp = []
        if type(res) == nltk.tree.Tree:
            for word in res:
                temp.append(word[0])
        if len(temp) >= 2:
            finish.append(" ".join(temp))
    return finish

ulasan_extract_train = opini_extractor(ulasan_tree_train)
all_opini_train = [item for sublist in ulasan_extract_train for item in sublist]

train_opini = readData(train_opini)
train_opini

def getDataframeAspect(dframe):
    new_dframe = dframe.replace({'positive':1, 'negative':1}).fillna(-1)
    return new_dframe

dframe_aspek_train = getDataframeAspect(train_opini)
x = dframe_aspek_train['opini'].to_list()
y = dframe_aspek_train.drop('opini', axis=1)

y_train_ambience = y['AMBIENCE#GENERAL'].to_numpy()
y_train_drink = y['DRINK#QUALITY'].to_numpy()
y_train_food = y['FOOD#QUALITY'].to_numpy()
y_train_restaurant = y['RESTAURANT#GENERAL'].to_numpy()
y_train_service = y['SERVICE#GENERAL'].to_numpy()

def getToken(ulasan):
    list_ulasan = ulasan
    token = [i.split() for i in list_ulasan]
    # print(token)
    all_token = sorted(list(set([item for sublist in token for item in sublist])))
    # all_token.pop(0)
    # print(all_token)
    return list_ulasan, token, all_token

def getTokenUji(ulasan, token):
    token_test = ulasan.split()
    token_test_filter = [tok for tok in token_test if tok in all_token]
    result_token = token + [token_test_filter]
    return result_token

list_ulasan, token, all_token = getToken(x)

def termWeighting(token, alltoken):
    termfreq = [[tok.count(alltok) for tok in token] for alltok in alltoken]
    docfreq = [sum(1 for tf in tfs if tf > 0 ) for tfs in termfreq]
    inversedf = [math.log10(len(tf) / df) for tf, df in zip(termfreq, docfreq)]
    tfxidf = [
                [(1 + math.log10(tf)) * inversedf if tf > 0 else tf for tf in termfreq]
                for termfreq, inversedf in zip(termfreq, inversedf)
            ]
    return inversedf, tfxidf

def termWeightingTest(new_token, alltoken, inversedf):
    termfreq = [[tok.count(alltok) for tok in new_token] for alltok in alltoken]
    tfxidf = [
                [(1 + math.log10(tf)) * inversedf if tf > 0 else tf for tf in termfreq]
                for termfreq, inversedf in zip(termfreq, inversedf)
            ]
    return tfxidf

inverse_docfreq, x_train = termWeighting(token, all_token)

lamb = 0.1
gamma = 0.001
C = 0.1
maxIter = 50

def getKernelLinear(data, weight):
    kernel_data = np.zeros((len(data), len(data)))

    for i in range(len(kernel_data)):
        for j in range(len(kernel_data)):
            jumlah = 0
            for k in range(len(weight)):
                jumlah += (weight[k][j] * weight[k][i])
            kernel_data[i][j] = jumlah
    return kernel_data

def getKernelLinearUji(data, datauji, weight):
    kernel_uji = np.zeros((len(data), len([datauji])))
    # print(kernel_uji.shape)
    w_transpose = np.array(weight).T
    # print(w_transpose)
    w_train = w_transpose[:-len([datauji])]
    # print(w_train)
    w_test = w_transpose[-len([datauji])]
    # print(w_test)

    for i in range(len(w_train)):
        jumlah = 0
        for j in range(len(w_test)):
            jumlah += w_test[j] * w_train[i][j]
        kernel_uji[i][0] = jumlah
    return kernel_uji

kernel = getKernelLinear(list_ulasan, x_train)

def getMatrikHessian(kernel, lamb, y_train):
    hessian = np.zeros(kernel.shape)
    for i in range(hessian.shape[0]):
        for j in range(hessian.shape[1]):
            hessian[i][j] = (y_train[i]*y_train[j]) * (kernel[i][j]+pow(lamb, 2))
    return hessian

def seqLearning(hessian, gamma, C, maxIter):
    alpha = np.zeros(hessian.shape[0])
    # print(alpha)
    error = np.zeros(hessian.shape[0])
    deltaError = np.zeros(hessian.shape[0])
    iter = 0

    while iter < maxIter:
        for i in range(hessian.shape[0]):
            error[i] = 0
            for j in range(hessian.shape[1]):
                error[i] += hessian[i][j] * alpha[i]
        for i in range(hessian.shape[0]):
            deltaError[i] = min(max((gamma * (1 - error[i])), -alpha[i]), C - alpha[i])
            alpha[i] = deltaError[i] + alpha[i]
        iter += 1 
    return alpha

def getBias(alpha, y_train, kernel):
    positif = alpha.tolist().index(max([data for idx, data in enumerate(alpha) if y_train[idx] == 1]))
    print("positif : ", positif)
    negatif = alpha.tolist().index(max([data for idx, data in enumerate(alpha) if y_train[idx] == -1]))
    print("negatif : ", negatif)
    kernel_pos = sum([alpha[i] * y_train[i] * kernel[i][positif] for i in range(len(y_train))])
    print("kernelpos : ", kernel_pos)
    kernel_neg = sum([alpha[i] * y_train[i] * kernel[i][negatif] for i in range(len(y_train))])
    print("kernelneg : ", kernel_neg)
    bias = -0.5*(kernel_pos + kernel_neg)
    return bias

def TrainingSVM(kernel, x_train, y_train, gamma, lamb, C, maxIter):

    hessian = getMatrikHessian(kernel, lamb, y_train)
    alpha = seqLearning(hessian, gamma, C, maxIter)
    bias = getBias(alpha, y_train, kernel)
    # print(alpha, bias)

    return alpha, bias

# For Aspect 

alpha_ambience, bias_ambience = TrainingSVM(kernel, x_train, y_train_ambience, gamma, lamb, C, maxIter)
alpha_drink, bias_drink = TrainingSVM(kernel, x_train, y_train_drink, gamma, lamb, C, maxIter)
alpha_food, bias_food = TrainingSVM(kernel, x_train, y_train_food, gamma, lamb, C, maxIter)
alpha_restaurant, bias_restaurant = TrainingSVM(kernel, x_train, y_train_restaurant, gamma, lamb, C, maxIter)
alpha_service, bias_service = TrainingSVM(kernel, x_train, y_train_service, gamma, lamb, C, maxIter)

y_sentiment = train_opini.drop('opini', axis=1).fillna(0)
y_sent = y_sentiment.values.tolist()
y_sent_merge = [item for sublist in y_sent for item in sublist]
y_sent_filter = [ele for ele in y_sent_merge if ele != 0]

y_train_sentiment = [1 if x == 'positive' else -1 for x in y_sent_filter]
# For Sentiment 

alpha_sentiment, bias_sentiment = TrainingSVM(kernel, x_train, y_train_sentiment, gamma, lamb, C, maxIter)

def testingSVM(alpha, bias, kerneluji, y_train):
    hasil = [] 

    for i in range(kerneluji.shape[1]):
        jumlah = 0
        for j in range(kerneluji.shape[0]):
            jumlah += alpha[j] * y_train[j] * kerneluji[j][i]
        hasil.append(jumlah)
    nilai_sentimen = [i+bias for i in hasil]
    kelas_aspek = [1 if i > 0 else -1 for i in nilai_sentimen]
    return kelas_aspek

def testingAspectOneData(alpha, bias, kerneluji, y_train):
    for i in range(kerneluji.shape[1]):
        jumlah = 0
        for j in range(kerneluji.shape[0]):
            jumlah += alpha[j] * y_train[j] * kerneluji[j][i]
    nilai_sentimen = jumlah + bias
    kelas_aspek = 1 if nilai_sentimen > 0 else -1 
    return kelas_aspek

def testingSentimentOneData(alpha, bias, kerneluji, y_train):
    for i in range(kerneluji.shape[1]):
        jumlah = 0
        for j in range(kerneluji.shape[0]):
            jumlah += alpha[j] * y_train[j] * kerneluji[j][i]
    nilai_sentimen = jumlah + bias
    kelas_sentimen = "positive" if nilai_sentimen > 0 else "negative"
    return kelas_sentimen

def getNameCategory(data):
    if(1 not in data):
        pass
    else:
        res = data.index(1)
        if(res == 0):
            return "AMBIENCE#GENERAL"
        if(res == 1):
            return "DRINK#QUALITY"
        if(res == 2):
            return "FOOD#QUALITY"
        if(res == 3):
            return "RESTAURANT#GENERAL"
        if(res == 4):
            return "SERVICE#GENERAL"

def getAspectSentimentTest(kernel_uji):
    aspect = []
    y_pred_ambience = testingAspectOneData(alpha_ambience, bias_ambience, kernel_uji, y_train_ambience)
    y_pred_drink = testingAspectOneData(alpha_drink, bias_drink, kernel_uji, y_train_drink)
    y_pred_food = testingAspectOneData(alpha_food, bias_food, kernel_uji, y_train_food)
    y_pred_restaurant = testingAspectOneData(alpha_restaurant, bias_restaurant, kernel_uji, y_train_restaurant)
    y_pred_service = testingAspectOneData(alpha_service, bias_service, kernel_uji, y_train_service)
    aspect.extend([y_pred_ambience, y_pred_drink, y_pred_food, y_pred_restaurant, y_pred_service])
    #   print(aspect)
    aspect_category_test = getNameCategory(aspect)
    sentiment_aspect = testingSentimentOneData(alpha_sentiment, bias_sentiment, kernel_uji, y_train_sentiment)
    result = {aspect_category_test: sentiment_aspect}
    #   print(result)
    return result

def testingDataUji(data_uji, token):
    aspect_sentiment_uji = []
    uji = [data_uji]
    uji_prepos = preprocessing(uji)
    uji_decontracted = decontracted(uji_prepos)
    uji_postag = postagUji(uji_decontracted)
    uji_tree = opini_rule_uji(uji_postag)
    uji_extract_opini = opini_extractor_uji(uji_tree)
    for opi in uji_extract_opini:
    #     print(opi)
        token_uji = getTokenUji(opi, token)
        tfidf_uji = termWeightingTest(token_uji, all_token, inverse_docfreq)
        # print(np.shape(tfidf_uji))
        kernel_uji = getKernelLinearUji(list_ulasan, opi, tfidf_uji)
        result = getAspectSentimentTest(kernel_uji)
        for key, val in result.items():
            if key != None:
                aspect_sentiment_uji.append(result)
    return aspect_sentiment_uji

# user_input = input("Enter your review about this restaurant: \n")
# # i generally like this fantastic place, it has amazing decor and makes everyone feel cozy. besides the food is good and comes in large portions but the wine is not worth with the price. and yeah one more, i think the manager should fire those staff with bad attitude

# hasil_absa = testingDataUji(user_input, token)
# print(hasil_absa)
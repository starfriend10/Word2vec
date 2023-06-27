# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 00:42:20 2021

@author: Junjie Zhu
"""


import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import SnowballStemmer

df = pd.read_csv('Your dataset.csv')

###--functions defined for preliminary treatment----------------------------------------
##define your own data preprocessing fucntions, and generate treated dataset
def word_processing(w):
    function1
    function2
    fucntion3
    ......
    return w

################################################################
import time
import itertools
import multiprocessing as mp
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
import gensim
from gensim.models.phrases import Phrases, Phraser
from sklearn.pipeline import Pipeline
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))


####
from itertools import combinations
lst_pairs = list(combinations(stop_words, 2))
lst_pairs2 = [(w[1], w[0]) for w in lst_pairs]

lst_w1 = [' '.join(s) for s in lst_pairs]
lst_w2 = [' '.join(s) for s in lst_pairs2]

stop_words2 = lst_w1 + lst_w2



df_kw0 = df.replace(np.nan, '')
lst_textti = [i for i in df_kw0['TI']] #title
lst_textkw = [i for i in df_kw0['DE']] #keywords
lst_textab = [i for i in df_kw0['AB']] #abstract

#treat title, abstract, and keywords differently
lst_text0 = [lst_textti[i]+'. '+lst_textab[i]+'##THIS IS A SPLIT POINT FOR KEYWORDS##'+lst_textkw[i] for i in range(len(lst_textti))]
lst_text = pd.Series(lst_text0)


#define the two hyperparameters based on preliminary tests
min_count = 3
threshold = 2


df_pw = pd.read_excel('Dataset of similar terms.xlsx')
pair_words = df_pw.iloc[:, 1:].values.tolist()

df_pw_dis = pd.read_excel('Dataset of opponent terms.xlsx')
pair_words_dis = df_pw_dis.iloc[:, 1:].values.tolist()

df_top = pd.read_csv('Dataset of the tokens to be included in the analysis.csv')
lst_topwords = df_top.iloc[:, 0].values.tolist()

parm_dict = {'n_feature':(100,150,200,250,300), 'window':(6,8,10,12,14),'epochs':(5,10,15,20,25,30),'alpha':(0.001,0.003,0.005,0.01)}


class word2vecmodel(BaseEstimator, TransformerMixin):

    def __init__(self, stopWords=None, lower=True, strip=True):
        self.lower      = lower
        self.strip      = strip
        self.stopWords  = stop_words
        self.stemmer    = stemmer

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [" ".join(sent) for sent in X]

    def text_prep(self, text):
        text0 = text.replace('-', ' ') #can also insert additional treatment
        text1 = re.sub(r'[^\w\s]', ' ', str(text0).lower().strip())
        text2 = text1.replace('industry 4 0', 'industry 4.0')
        text3 = " ".join([w for w in text2.split()])
        return text3

    def lstwords(self, i):
        lst1 = self.lst_words_all1[i]
        lst_words1 = [word_processing(w) for w in lst1 if w not in stop_words] #define your word_processing function to preprocess textual data
        lst_words2 = [w for w in lst_words1 if (w not in stop_words) and (any(letter.isalpha() for letter in w))]
        lst_words3 = [w for w in lst_words2 if w not in stop_words2]        
        lst_words4 = [self.stemmer.stem(w) for w in lst_words3]
        return lst_words4
    
    def words_similarity(self, model, pair_words):
        lst_scores = []
        for i in range(len(pair_words)):
            word1 = pair_words[i][0]
            word2 = pair_words[i][1]        
            score = model.wv.similarity(word1, word2)
            lst_scores.append(score)
        avg_score = np.mean(lst_scores)
        return lst_scores, avg_score

    def retrievability(self, model, lst_topwords):
        w2v_words = list(model.wv.key_to_index)
        lst_retr = [w for w in lst_topwords if w in w2v_words]
        lst_notretr = [w for w in lst_topwords if not w in w2v_words]
        rtb = len(lst_retr)*100/len(lst_topwords)

        return rtb, lst_notretr

    def kwdata_split(self, X):
        X1 = [sent.split('##THIS IS A SPLIT POINT FOR KEYWORDS##')[0] for sent in X]
        X2 = pd.Series(X1)        
        KW = [[sent.split('##THIS IS A SPLIT POINT FOR KEYWORDS##')[1]] for sent in X]
        KW2 = [sent[0].split('; ') for sent in KW]
        
        return X2, KW2
    
    def modeling(self, X, n_feature, window, epochs, alpha):
        X,KW = self.kwdata_split(X)

        sentence_stream = [[i for i in word_tokenize(self.text_prep(sent))] for sent in X]        
        bigram = Phrases(sentence_stream, min_count=min_count, threshold=threshold, delimiter=' ')
        bigram_phraser = Phraser(bigram)
        tokens_ = bigram_phraser[sentence_stream]
      
        trigram = Phrases(tokens_, min_count=min_count, threshold=threshold, delimiter=' ')
        trigram_phraser = Phraser(trigram)
        tokens__ = trigram_phraser[tokens_]

        quadgram = Phrases(tokens__, min_count=min_count, threshold=threshold, delimiter=' ')        

        lst_words_quad = [[t for t in quadgram[trigram[bigram[sent]]]] for sent in sentence_stream]
        
        lst_words_all0 = [kw+lst_w for kw, lst_w in zip(KW,lst_words_quad)]
        self.lst_words_all1 = [[e for e in lst if e] for lst in lst_words_all0]

        if __name__=="__main__":
            mp.freeze_support() # optional if the program is not frozen
            # start processes
            pool = mp.Pool(20) # use 20 CPU cores
            output = pool.map(self.lstwords, range(len(sentence_stream)))
            pool.close()
            pool.join()
        
        lst_words_all = [e for e in output]

        lst_words_all2 = [w for w in lst_words_all if w]
        
        tokenize = pd.Series(lst_words_all2)
        w2vec_model=gensim.models.Word2Vec(tokenize, workers = 20, min_count = min_count, vector_size = n_feature, window = window, 
                                           sg = 1, alpha = alpha)
        w2vec_model.train(tokenize, total_examples= len(X), epochs=epochs)
        
        return w2vec_model

    def transform(self, X):
        n_feature, window, epochs, alpha = [tup for k,tup in parm_dict.items()]
        parm_combo = list(itertools.product(n_feature, window, epochs, alpha))
    
        lst_loss1 = []
        lsts_scores1 = []
        lst_loss2 = []
        lsts_scores2 = []
        lst_parms = []
        lst_rtb = []
        lsts_notretr = []
        n = 0
        for parms in parm_combo:
            n_feature, window, epochs, alpha = parms
            
            timex = time.time()
            
            w2vec_model = self.modeling(X, n_feature, window, epochs, alpha)
            
            lst_scores1, avg_score1 = self.words_similarity(w2vec_model, pair_words)
            lst_loss1.append(avg_score1)
            lsts_scores1.append(lst_scores1)
            
            lst_scores2, avg_score2 = self.words_similarity(w2vec_model, pair_words_dis)
            lst_loss2.append(avg_score2)
            lsts_scores2.append(lst_scores2)            
            
            lst_parms.append([n_feature, window, epochs, alpha])

            rtb, lst_notretr = self.retrievability(w2vec_model, lst_topwords)
            lst_rtb.append(rtb)
            lsts_notretr.append(lst_notretr)
            
            print("Parameter combination: "+ str(parms)+
                  "; Average similarity for pairs 1 is: "+str(avg_score1)+
                  "; Average similarity for pairs 2 is: "+str(avg_score2)+
                  "; Retrievability is: "+str(rtb)+"%"+
                  "; Not retrievable: "+str(lst_notretr))
            
            
            timey = time.time()
            timeusedx = timey - timex
            timeusedy = timey - time1            
            print("This iteration's computing time is "+str(timeusedx/3600)+' hrs. '+
                  'The cumulative computing time is '+str(timeusedy/3600)+' hrs')
            
            n += 1            
            if n%90 == 0 and n>0:
                print('Sleeping for 300 seconds.........................................') #giving the machine a time for cooling down, then restart it
                time.sleep(300)
        
        return lst_loss1, lsts_scores1, lst_loss2, lsts_scores2, lst_parms, lst_rtb, lsts_notretr


time1 = time.time()
preprocess_all = Pipeline([('word2vec', word2vecmodel())])
lst_loss1, lsts_scores1, lst_loss2, lsts_scores2, lst_parms, lst_rtb, lsts_notretr = preprocess_all.fit_transform(lst_text)
time2 = time.time()
timeused = time2 - time1
print('All computing time is '+str(timeused/3600)+' hrs')





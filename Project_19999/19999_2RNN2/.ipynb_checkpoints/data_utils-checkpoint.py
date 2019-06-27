import os
import cv2
import numpy as np
import pandas as pd
from glob import glob

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf
import sys
import pandas as pd
import jieba
import jieba.posseg as pseg
from time import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import gensim
from tqdm import tqdm_notebook
import re
from gensim.models import word2vec
import random
import warnings
import json
from keras.utils import np_utils


from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
warnings.filterwarnings('ignore')

"""
資料夾及檔案路徑變數
"""

"""放置全部資料集的資料夾"""


# reload(sys)
# sys.setdefaultencoding('utf-8')



class JiebaSegmentor:

    def __init__(self, dict_path, userdict=[], stopwords=False, stopwords_path=None):
        self.dict_path = dict_path
        self.userdict = userdict
        self.dictionary_init()
        self.stopwords_path = stopwords_path
        self.stopwords = stopwords
        self.stopwords_set = set()
        self.stopwords_init()

    def dictionary_init(self):
        jieba.set_dictionary(self.dict_path)
        for path in self.userdict:
            jieba.load_userdict(path)

    def stopwords_init(self):
        if self.stopwords_path:
            with open(self.stopwords_path, 'r') as stopwords:
                for stopword in stopwords:
                    self.stopwords_set.add(stopword.strip('\n'))

    def taiwan_country(self):
        return [u'臺北', u'台北', u'基隆', u'臺中', u'台中', u'臺南', u'台南', u'高雄',
                u'宜蘭', u'桃園', u'新竹', u'苗栗', u'彰化', u'南投', u'嘉義', u'雲林',
                u'屏東', u'臺東', u'台東', u'花蓮', u'澎湖']

    def wordToNumber(self, input_text):

        target = u''
        for s in input_text:

            if (s == u'零') or (s == '0'):
                to_word = u'0'
            elif (s == u'一') or (s == u'壹') or (s == '1'):
                to_word = u'1'
            elif (s == u'二') or (s == u'兩') or (s == u'貳') or (s == '2'):
                to_word = u'2'
            elif (s == u'三') or (s == u'參') or (s == '3'):
                to_word = u'3'
            elif (s == u'四') or (s == u'肆') or (s == '4'):
                to_word = u'4'
            elif (s == u'五') or (s == u'伍') or (s == '5'):
                to_word = u'5'
            elif (s == u'六') or (s == u'陸') or (s == '6'):
                to_word = u'6'
            elif (s == u'七') or (s == u'柒') or (s == '7'):
                to_word = u'7'
            elif (s == u'八') or (s == u'捌') or (s == '8'):
                to_word = u'8'
            elif (s == u'九') or (s == u'玖') or (s == '9'):
                to_word = u'9'
            else:
                to_word = s

        target = target + to_word
        return target

    def input_text_preprocessing(self, input_text):

#         if type(input_text) is not unicode:
#             input_text = input_text.decode('utf-8')

        #         input_text = self.wordToNumber(input_text)
        return input_text

    def get_names(self, input_text):
        """
        取得姓名
        :param input_text:
        :return:
        """

        input_text = self.input_text_preprocessing(input_text)
        names = []
        words = pseg.cut(input_text)
        print (words)
        for w, f in words:
            if f.lower() == 'nr':
                names.append(w)
        for name in names:
            print (name)
        return names

    def lcut(self, input_text, return_type='pandas'):
        """
        斷詞
        :param input_text:
        :param return_type:
        :return: pandas
        """

        input_text = self.input_text_preprocessing(input_text)
        cut_raw = jieba.lcut(input_text)
        key = []

        for k in cut_raw:
            if self.stopwords:
                if k in self.stopwords_set:
                    continue

            key.append(k)

        result = pd.DataFrame({"word": key})
        if return_type == 'pandas':
            return result
        elif return_type == 'dict':
            return result.to_dict('index').values()
        else:
            return result

    def pseg_lcut(self, input_text, return_type='pandas'):
        """
        斷詞+詞性
        :param input_text:
        :param return_type:
        :return: pandas
        """

        input_text = self.input_text_preprocessing(input_text)
        cut_raw = pseg.lcut(input_text)
        key = []
        value = []

        for k, v in cut_raw:
            tag = v
            if self.stopwords:
                if k in self.stopwords_set:
                    continue

            if k in self.taiwan_country():
                tag = u'ns'
            if len(k) > 1 and tag == u'x':
                tag = u'n'
            key.append(k)
            value.append(tag)

        result = pd.DataFrame({"word": key, "tag": value})
        if return_type == 'pandas':
            return result
        elif return_type == 'dict':
            return result.to_dict('index').values()
        else:
            return result

    def pseg_lcut_combie_num_eng(self, input_text, return_type='pandas'):
        """
        將數字與英文結合成同一欄位
        :param input_text:
        :param return_type:
        :return: pandas
        """

        input_text = self.input_text_preprocessing(input_text)
        seg_pd = self.pseg_lcut(input_text)
        seg_dict_list = []
        m_eng_list = []
        CombieTuple = namedtuple('CombieTuple', {
            'index',
            'word',
            'sp'})

        for index, seg in seg_pd.iterrows():
            #     print type(seg)
            #     print seg
            seg_dict = {
                "word": seg['word'],
                "sp": seg['tag']
            }

            if seg['tag'] == 'm':
                #         m_eng_dict.update(seg_dict)
                combie_tuple = CombieTuple(
                    index=index,
                    word=seg['word'],
                    sp=seg['tag']
                )
                m_eng_list.append(combie_tuple)
            #             continue

            if seg['tag'] == 'eng':
                if m_eng_list:
                    if m_eng_list[0].index + 1 == index:
                        seg_dict = {
                            "word": m_eng_list[0].word + seg['word'],
                            "sp": m_eng_list[0].sp + '+' + seg['tag']
                        }
                        m_eng_list = []
                        del seg_dict_list[index - 1]

            seg_dict_list.append(seg_dict)

        if return_type == 'pandas':
            return pd.DataFrame(seg_dict_list)
        elif return_type == 'dict':
            return seg_dict_list
        else:
            return pd.DataFrame(seg_dict_list)



def cut_to_word(s,js):
    w_list = js.lcut(s, cut_type='list')
    combie = ''
    combieNO = ''
    for i,w in enumerate(w_list):
        if w == ' ':
            continue
        if len(w)<=1:
            continue
        conditionWord = str(combieNO) + str(w)
        if combieNO != '': #and len(conditionWord) == 4:
            #combie = combie + w +','
            combie = combie+combieNO+w
            #print(combieNO+w)
        else:
            combie = combie + w
            
        if i < len(w_list) - 1:
            combie = combie + ','
        if w in ['無法','安裝','編輯','刪除','新增']:
            combieNO = w
        else:
            combieNO = ''
    return combie    

def cut_to_word_pandas(s,js):
    w_pandas = js.lcut(s, return_type='pandas')
    w_list = list(w_pandas['word'])
    combie = ''
    combieNO=''
    for i,w in enumerate(w_list):
        if w is ' ':
            continue
        conditionWord = str(combieNO) + str(w)
        if combieNO != '': #and len(conditionWord) == 4:
            #combie = combie + w +','
            combie = combie+combieNO+w
            #print(combieNO+w)
        else:
            combie = combie + w
            
        if i < len(w_list) - 1:
            combie = combie + ','
        if w in ['無法','安裝','編輯','刪除','新增']:
            combieNO = w
        else:
            combieNO = ''
    return combie  

def produce_cbowfile(data,jieba_stopwords_path):
    sentences = []
    stop_words = open(jieba_stopwords_path, encoding="utf-8").read().splitlines()

    for i, text in enumerate(tqdm_notebook(data)):
        line = []

        for w in jieba.cut(text, cut_all=False):

            ## remove stopwords and digits
            ## can define your own rules
            if w == ' ':
                continue
            if len(w)<=1:
                continue
            if w not in stop_words and not bool(re.match('[0-9]+', w)):
                ##print(w)
                if w == '流程' :
                    line.append('簽核流程')
                if w == '服務' :
                    line.append('相關服務')
                else:
                    line.append(w)

        sentences.append(line)
    model = word2vec.Word2Vec(size=256, min_count=5, window=5, sg=0, negative=15, iter=10)
    model.build_vocab(sentences)
    for i in range(20):
        random.shuffle(sentences)
    model.train(sentences, total_examples=len(sentences), epochs=1)
    model.save('1999_CBOW')
    ## print an example
    #model.wv['系統']
    return sentences,model

def get_embedding_martrix():
    w2v = word2vec.Word2Vec.load('1999_CBOW')
    word2id = {k:i for i, k in enumerate(w2v.wv.vocab.keys())}
    id2word = {i:k for k, i in word2id.items()}
    words_len = len(word2id)
    #print(word2id.items())
    embedding = np.zeros((words_len+1, 256))
    for k, v in word2id.items():
        embedding[v] = w2v.wv[k]
        ##print(k)
    return embedding

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=256):
    
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments.apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)


def covert_sequences(data,NUM_WORDS, MAX_LEN):
    #https://zhuanlan.zhihu.com/p/59257654
    # 標點符號過濾
    #WORD_FILTERS = '!"#$&()*+,-./:;<=>?@[\\]^_{|}~\t\n'
    
    token_intent = Tokenizer(#filters=WORD_FILTERS,
                            num_words=NUM_WORDS,
                            split=",")#创建分词器，设置为只考虑前NUM_WORDS个最常见的单词
    
    # 斷詞後全部合併丟到fit_on_texts,組出字典
    token_intent.fit_on_texts(data)##构建单词索引
    
    #將文字轉為數字序列
    train_seq_intent = token_intent.texts_to_sequences(data)##将字符串转换为整数索引组成的列表
    
    train_word_index = token_intent.word_index#找回单词索引
    print('Found %s unique tokens.' % len(train_word_index))
    #print(word_index)
    # 截長補短，讓所有影評所產生的數字序列長度一樣
    returndata = sequence.pad_sequences(train_seq_intent, maxlen=MAX_LEN)#不仅能使用二进制，也支持one-hot外的向量化编码模式
    # print train_data_intent.shape
    return returndata,train_word_index

    
def data_reduction(df):
    df = df.drop( labels = df[df['description'].str.len()<5].index, axis = 0 )
    #[(df['category_a_target']!= 2) | (df['category_a_target']== 4) | (df['category_a_target']== 5)
    df = df.drop( labels = df[(df['category_a_target']!= 6)].index, axis = 0 )
    df.head(20)
    # 資料打散
    #注意要打乱数据，因为原始数据是分类排好序的
    indices = np.arange(df.shape[0]) 
    np.random.shuffle(indices)
    #df = df.sample(frac=1).reset_index(drop=True)
    df.head(20)
    return df
    
def seperatedata(x,y):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=42)
    return x_train,x_test,y_train,y_test

def labelEncoding(data):
    # label 做 onehot
    y_one_hot = np_utils.to_categorical(data)
    print (y_one_hot[0])
    # 總共類別數
    num_classes = len((y_one_hot[0]))
    print ('num_classes = {}'.format(num_classes))
    return y_one_hot,num_classes

import decimal 

def float_to_str(f): 
    # create a new context for this task 
    ctx = decimal.Context() 
    # 20 digits should be enough for everyone  
    ctx.prec = 10 
    """ 
    Convert the given float to a string 
    without resorting to scientific notation 
    """ 
    d1 = ctx.create_decimal(repr(f)) 
    return format(d1,'f')


def predict(model,data,mapping):
#     model = load_model(model_path)
    y_predict_probability = model.predict(data)
    y_predict = model.predict_classes(data)
    print(len(mapping))
    def to_cat_name(x): 
        print(x)
        return mapping[x]
    
    return_Name=to_cat_name(y_predict)
    
    predict_arr = [] 
    for row in y_predict_probability: 
        row_arr = [] 
        classIndex=0
        for item in row:
            #print(str(mapping[classIndex])+':'+str(float_to_str(item)))
            row_arr.append(str(mapping[classIndex])+':'+str(float_to_str(item))) #mapping[classIndex]+':'+
            classIndex=classIndex+1
        predict_arr.append(row_arr) 
    
    return to_cat_name(y_predict), y_predict, predict_arr,y_predict_probability

def predict_class(model,data,batchsize,mapping):
    y_predict_probability = model.predict(data, batch_size=batchsize, verbose=1)
    predict_arr = []
    predictClass = []
    for row in y_predict_probability: 
            row_arr = [] 
            classIndex=0
            selectClass=0
            selectProbability=0
            for item in row: 
                # print(float_to_str(item)) 
                row_arr.append(mapping[classIndex]+':'+float_to_str(item)) 
                #print(float(item))
                if(selectProbability <= float(item)):
                    #print(classIndex)
                    selectProbability=float(item)
                    selectClass=classIndex
                classIndex=classIndex+1
            predict_arr.append(row_arr) 
            predictClass.append(selectClass)
            
    return mapping[predictClass],predictClass,predict_arr,y_predict_probability
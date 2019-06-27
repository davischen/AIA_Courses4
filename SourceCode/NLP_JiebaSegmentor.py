# -*- coding: utf-8 -*-
import sys
import os

# reload(sys)
# sys.setdefaultencoding('utf-8')

from collections import namedtuple
import pandas as pd
import jieba
import jieba.posseg as pseg


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
                    self.stopwords_set.add(stopword.strip('\n').decode('utf-8'))

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

        if type(input_text) is not unicode:
            input_text = input_text.decode('utf-8')

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
            print (name.encode('utf-8'))
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

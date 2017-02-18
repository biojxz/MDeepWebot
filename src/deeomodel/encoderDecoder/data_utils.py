#coding:utf-8
'''
 Data Loader
'''
import logging
import os
import random
import numpy as np

logging.basicConfig(level=logging.DEBUG)
class Corpus:
    NULL = 'NULL'
    PADDING = 'PADDING'
    END = 'END'
    GO = 'GO'

    def __init__(self,working_path,batch_size,sentence_size,total_set=10000,test_set_size = 1000):
        self.test_size_size = test_set_size
        self.working_path = working_path + '/'
        logging.debug('Corpus working path : %s' % working_path)
        self.batch_Size = batch_size
        self.sentence_size = sentence_size
        logging.info('Corpus Batch_size=%d,Sentence_Size(time_steps)=%d' % (batch_size,sentence_size))
        self.pointer = 0
        input1 = open(self.working_path+'stc_weibo_train_post','r')
        input2 = open(self.working_path+'stc_weibo_train_response','r')
        self.raw_corpus = list()
        self.dicts = dict()
        self.tfcounter = dict()
        last_sentence = ''
        for i in range(total_set):
            post = input1.readline().strip('\n').strip('\r').decode('utf-8').split()
            response = input2.readline().strip('\n').strip('\r').decode('utf-8').split()
            if len(post) > 1 and len(response) > 1:
                self.raw_corpus.append((post,response))
                for char in post:
                    if char not in self.dicts:
                        self.dicts[char] = len(self.dicts)
                        self.tfcounter[char] = 0
                    self.tfcounter[char] = self.tfcounter[char] + 1

                for char in response:
                    if char not in self.dicts:
                        self.dicts[char] = len(self.dicts)
                        self.tfcounter[char] = 0
                    self.tfcounter[char] = self.tfcounter[char] + 1
            else:
                total_set = total_set + 1 #1000

        logging.debug('Corpus Total Number of Raw QA Pairs : %d' % len(self.raw_corpus))

    def record_to_file(self,filepath='mydict.txt'):
        logging.debug('Corpus Ready to write dictionary to the position : %s ' % filepath)
        file = open(self.working_path+'mydict.txt','w+')
        for id in self.token_dict:
            file.write('id=%d\ttf=%d\t%s\n' %(self.token_dict[id],self.tfcounter.get(id,0),id.encode('utf-8')))
        logging.info('Corpus Already wrote dictionary to the position : %s ' % filepath)

    def sentenceTotoken(self,max_token_size = 1000):
        # test_set
        test_selected_rate = self.test_size_size / (0.0 + len(self.raw_corpus))
        self.test_x = []
        self.test_y = []
        #sort
        max_token_size = min(max_token_size - 4,len(self.dicts))
        STOP_WORDS = ['，', '。', '＂', '！', '？']
        for item in STOP_WORDS:
            self.tfcounter[item.decode('utf-8')] = -1000000
        sorted_dict = sorted(self.tfcounter.items(),key=lambda x:x[1],reverse=True)
        valid_words = sorted_dict[0:max_token_size]
        #new dict
        self.token_dict = dict()
        self.token_dict[self.NULL] = 0
        self.token_dict[self.PADDING] = 1
        self.token_dict[self.GO] = 2
        self.token_dict[self.END] = 3

        self.id_dict = list()
        self.id_dict.append(self.NULL)
        self.id_dict.append(self.PADDING)
        self.id_dict.append(self.GO)
        self.id_dict.append(self.END)
        for item in valid_words:
            if self.tfcounter.get(item[0]) >= 0:
                self.token_dict[item[0]] = len(self.token_dict)
                self.id_dict.append(item[0])
        #sentence2ids
        self.id_corpus = list()
        for pair in self.raw_corpus:
            context = pair[0]
            answer = pair[1]
            context_ids = [self.token_dict.get(word,0) for word in context]
            answer_ids = [self.token_dict.get(word,0) for word in answer]
            if len(context_ids) > self.sentence_size:
                context_ids = context_ids[0:self.sentence_size]
                context_ids[self.sentence_size-1] = 3
            if len(answer_ids) > self.sentence_size:
                answer_ids = answer_ids[0:self.sentence_size]
                answer_ids[self.sentence_size-1] = 3
            if len(context_ids) == self.sentence_size:
                context_ids[self.sentence_size - 1] = 3
            if len(answer_ids) == self.sentence_size:
                answer_ids[self.sentence_size - 1] = 3
            if len(context_ids) < self.sentence_size:
                context_ids.append(3)
                while len(context_ids) < self.sentence_size:
                    context_ids.append(1)
            if len(answer_ids) < self.sentence_size:
                answer_ids.append(3)
                while len(answer_ids) < self.sentence_size:
                    answer_ids.append(1)
            if random.random() < test_selected_rate:
                self.test_x.append(context_ids)
                self.test_y.append(answer_ids)
            self.id_corpus.append((context_ids,answer_ids))
        logging.info('Test Set Size:  %d' % len(self.test_x))
        self.record_to_file()

    def get_vocab_size(self):
        return len(self.token_dict)


    def get_decoder_inputs(self):
        dxs = []
        for i in range(0,self.batch_Size):
            x = np.zeros([self.sentence_size])
            x[0] = 3 # GO
            dxs.append(x)
        return np.transpose(dxs)

    def next_batch(self):
        #[batch_size * sequence_len]
        xs = []
        ys = []
        weights = []
        for i in range(self.pointer,self.pointer + self.batch_Size):
            index = i % len(self.id_corpus)
            xs.append(self.id_corpus[index][0])
            ys.append(self.id_corpus[index][1])
            weight = []
            for y in self.id_corpus[index][1]:
                if y == 1 or y == 2:
                    weight.append(0)
                elif y == 0:
                    weight.append(0)
                else:
                    weight.append(1)
            weights.append(weight)
        self.pointer = (self.pointer + self.batch_Size) % len(self.id_corpus)
        return  np.transpose(xs),np.transpose(ys),np.transpose(weights)

    def get_test_batch(self):
        return self.test_x,self.test_y


if __name__ == '__main__':
    corpus = Corpus(os.path.abspath('')+'/',10,10)
    corpus.sentenceTotoken(1000)
    a=corpus.next_batch()
    b=corpus.next_batch()
    c=corpus.next_batch()
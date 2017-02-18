#coding:utf-8
from real_seq2seq import Model
from model_configs import HParam
import jieba
import tensorflow as tf
import numpy as np
import logging
import os

class TalkAPI():
    def __init__(self,param,workspace,dictspace):
        self.param = param
        self.token2id = dict()
        self.id2token = dict()
        # Load trained dicts from file
        input_file = open(dictspace+'/mydict.txt')
        for line in input_file.readlines():
            parts = line.strip('\n').split('\t')
            id = int(parts[0][3:])
            token = parts[2].decode('utf-8')
            self.token2id[token] = id
            self.id2token[id] = token
        # load trained Model
        param = HParam()
        INPUT_SHAPE = [param.batch_size]
        OUTPUT_SHAPE = [param.batch_size]
        VOCAB_SIZE = len(self.token2id)
        self.model = Model(INPUT_SHAPE, OUTPUT_SHAPE, 'lstm', param.rnn_cell_layer, param.rnn_cell_size, param.batch_size,
                      VOCAB_SIZE, VOCAB_SIZE, param.embedding_len)

        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(workspace)
        logging.info('Checkpoint path: %s' % os.path.abspath('workspace'))
        if ckpt and ckpt.model_checkpoint_path:
            logging.info('Session is ready to load the Checkpoint: %s' % ckpt.model_checkpoint_path)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
            self.sess = sess
            logging.info('Session has loaded the Checkpoint: %s' % ckpt.model_checkpoint_path)
        else:
            last_step = -1
            raise Exception('None checkpoint has been found !   '+workspace)
            logging.info('No Checkpoint founded!')

    def talk(self,sentence):
        xs,ys,weights,decoder_xs = self.sentenceToBatch(sentence)
        feed_dict = {  self.model.learning_rate:0.45}

        for i, sxs in enumerate(self.model.xs):
            feed_dict[sxs.name] = xs[i]
        for i, sys in enumerate(self.model.ys):
            feed_dict[sys.name] = ys[i]
        for i, sys in enumerate(self.model.decoder_xs):
            feed_dict[sys.name] = decoder_xs[i]
        for i, sys in enumerate(self.model.weights):
            feed_dict[sys.name] = weights[i]
        outputs = self.sess.run([self.model.op_output],feed_dict=feed_dict)[0]
        return self.outputsToSentence(outputs)

    def sentenceToBatch(self,sentence):
        xs = []
        ys = []
        weights = []
        x = []
        y = range(0,10)
        weight = range(0,10)
        for word in jieba.lcut(sentence):
            x.append(self.token2id.get(word,0))
        if len(x) < self.param.sequence_len:
            x.append(self.token2id.get('END'))
            while len(x) < self.param.sequence_len:
                x.append(self.token2id.get('PADDING'))
        elif len(x) == self.param.sequence_len:
            x[self.param.sequence_len - 1] = self.token2id.get('END')
        elif len(x) > self.param.sequence_len > self.param.sequence_len:
            x = x[0:9]
            x.append(self.token2id.get('END'))
        for i in range(self.param.batch_size):
            xs.append(x)
            ys.append(y)
            weights.append(weight)
        dxs = []
        for i in range(0, self.param.batch_size):
            x = np.zeros([self.param.sequence_len])
            x[0] = 3  # GO
            dxs.append(x)
        return np.transpose(xs),np.transpose(ys),np.transpose(weights),np.transpose(dxs)

    def outputsToSentence(self,outputs):
        line = ''
        for ids in range(self.param.sequence_len):
            id = np.argmax(outputs[ids][0])
            line = line + self.id2token.get(id,'XX')
        line = line
        return line
param = HParam()
api = TalkAPI(param, workspace=param.test_workspace,
                  dictspace=param.test_dictspace)
def test():
    input_text = raw_input('Say:')
    while(input_text!='END'):
        response = api.talk(input_text.decode('utf-8'))
        print response.encode('utf-8')
        input_text = raw_input('Say:')


if __name__ == '__main__':
    # param = HParam()
    # run(param)
    test()
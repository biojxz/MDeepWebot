#coding:utf-8
import tensorflow as tf
import numpy as np
from data_utils import Corpus
from model_configs import HParam
import logging
from seq2seq_model import Model
import time
# http://www.cnblogs.com/edwardbi/p/5559338.html

class EDModel:
    def __init__(self,param):
        '''

        :param param:
        '''
        self.param = param
        #首先需要加载语料库
        corpus = Corpus(param.train_dictspace, param.batch_size, param.sequence_len, param.max_train_sets)
        corpus.sentenceTotoken(param.dict_size)
        self.corpus = corpus
        #定义具体的模型
        INPUT_SHAPE = [param.batch_size]
        OUTPUT_SHAPE = [param.batch_size]
        VOCAB_SIZE = corpus.get_vocab_size()
        self.model = Model(INPUT_SHAPE, OUTPUT_SHAPE, 'lstm', param.rnn_cell_layer, param.rnn_cell_size, param.batch_size,
                      VOCAB_SIZE, VOCAB_SIZE, param.embedding_len, param.sequence_len, param.sampled_size)
        self.saver = tf.train.Saver()
        self.global_steps = tf.get_variable(name='global_steps', dtype=tf.int32,
                                       initializer=tf.zeros(shape=[1], dtype=tf.int32))
        #初始化Session，并且进行加载
        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(param.train_workspace)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.last_step = int(ckpt.model_checkpoint_path.split('-')[-1])
            logging.info('Session has loaded the Checkpoint: %s' % ckpt.model_checkpoint_path)
        else:
            self.last_step = -1
            self.sess.run(tf.global_variables_initializer())
            logging.info('No Checkpoint founded!')
        #固定的Decoder输入
        self.decoder_xs = corpus.get_decoder_inputs()
        self.last_timestamp = time.time()

    def train_step(self,step):
        xs, ys, weights = self.corpus.next_batch()
        step_lr = max(param.min_lr, param.lr * np.power(param.lr_decay, step / param.record_intervals))
        feed_dict = {
            self.model.learning_rate: step_lr
        }
        for i, sxs in enumerate(self.model.xs):
            feed_dict[sxs.name] = xs[i]
        for i, sys in enumerate(self.model.ys):
            feed_dict[sys.name] = ys[i]
        for i, sys in enumerate(self.model.decoder_xs):
            feed_dict[sys.name] = self.decoder_xs[i]
        for i, sys in enumerate(self.model.weights):
            feed_dict[sys.name] = weights[i]
        loss, _, output = self.sess.run([self.model.cost, self.model.train_op, self.model.output], feed_dict=feed_dict)
        return step_lr,loss,_,output



    def run(self):
        param = self.param
        for step in range(self.last_step+1,param.training_steps):
            xs,ys,weights = self.corpus.next_batch()
            step_lr,loss,_,output = self.train_step(step)
            if step % param.record_intervals == 0:
                logging.info(
                    'Learning Rate: %f' % (step_lr))
                print 'step %d ---- loss:%s,time consuming:%.2fs' % (step,loss,time.time()-self.last_timestamp)
                self.last_step = time.time()
                ys = np.transpose(ys)
                for i in range(10):
                    line = ''
                    for ids in range(8):
                        id = np.argmax(output[ids][i])
                        line = line + self.corpus.id_dict[id]
                    line = line + '\t'
                    for id in ys[i]:
                        line = line + self.corpus.id_dict[id]
                    print line.encode('utf-8')
                save_path = self.saver.save(self.sess, param.train_workspace+'/save_net.ckpt',global_step=step)
                print("Save to path: ", save_path)
            self.global_steps = self.global_steps + 1

    print 'hello world'


if __name__ == '__main__':
    param = HParam()
    EDModel(param).run()

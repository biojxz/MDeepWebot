#coding:utf-8
import tensorflow as tf
import numpy as np
from data_utils import Corpus
from model_configs import HParam
import os
import logging
import jieba
# http://www.cnblogs.com/edwardbi/p/5559338.html


class Model:
    def __init__(self,input_shape,output_shape,cell_type,cell_layer,cell_size,batch_size,input_vocab_size,output_vocab_size,embedding_size,seq_len=10,sampled_nums=1024):
        self.output_vocab_size = output_vocab_size
        self.batch_size = batch_size
        self.output_shape = output_shape
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.cell = cell_type
        self.cell_layer = cell_layer
        self.cell_size = cell_size
        self.embedding_size = embedding_size
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.sampled_nums = sampled_nums
        with tf.name_scope('inputs'):
            self.xs = []
            self.ys = []
            self.decoder_xs = []
            self.weights = []
            for i in range(seq_len):
                self.xs.append(tf.placeholder(dtype=tf.int32,shape=[batch_size],name="inputs{0}".format(i)))
                self.weights.append(tf.placeholder(dtype=tf.float32,shape=[batch_size],name="weights{0}".format(i)))
                self.ys.append(tf.placeholder(dtype=tf.int32,shape=[batch_size],name="outputs{0}".format(i)))
                self.decoder_xs.append(tf.placeholder(dtype=tf.int32,shape=output_shape,name="decoder_xs{0}".format(i)))
        with tf.name_scope('rnn'):
            self.add_rnn_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.learning_rate = tf.Variable(0,dtype=tf.float32,name='learning_rate')
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def add_rnn_layer(self):
        if self.cell == 'lstm' or self.cell != 'lstm':
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
            if self.cell_layer is not None and self.cell_layer > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([cell]*self.cell_layer,state_is_tuple=True)
            self.rnn_init_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        if self.cell == 'gru':
            cell = tf.nn.rnn_cell.GRUCell(self.cell_size)
            self.rnn_init_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        output_projection = None
        self.softmax_loss_function = None
        if self.sampled_nums > 0 and self.sampled_nums < self.output_vocab_size:
            w = tf.get_variable(name='output_proj_w',shape=[self.cell_size,self.output_vocab_size])
            w_t = tf.transpose(w)
            b = tf.get_variable("proj_b", [self.output_vocab_size])
            output_projection = (w,b)
            def sampled_loss(inputs, labels):
                with tf.device("/cpu:0"):
                    labels = tf.reshape(labels, [-1, 1])
                    return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, self.sampled_nums,
                                                      self.output_vocab_size)
            self.softmax_loss_function = sampled_loss
            logging.info('Using output projection')
        else:
            logging.info('Not using output projection')

        self.output, self.state = tf.nn.seq2seq.embedding_attention_seq2seq(self.xs, self.decoder_xs, cell, self.input_vocab_size,
                                                                            self.output_vocab_size, self.embedding_size,
                                                                       feed_previous=True,output_projection=output_projection,scope='embedding_attention_seq2seq')
        if output_projection is not None:
            self.output= [
                tf.matmul(output, output_projection[0]) + output_projection[1]
                for output in self.output
                ]
            self.op_output =self.output


    def compute_cost(self):
        # add the right weights that 0 denotes the null/poadding
        self.loss = tf.nn.seq2seq.sequence_loss_by_example(
            logits=self.output,
            targets=self.ys,
            weights=self.weights,
            name='loss',
            #softmax_loss_function=self.softmax_loss_function
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(self.loss,name='loss_sum'),
                self.batch_size,
                name='batch_cost'
            )
            tf.scalar_summary('cost',self.cost)

    def ms_error(self, y_prediction, y_target):
        return tf.square(tf.sub(y_prediction,y_target))

def run(param,isTrain=True):
    corpus = Corpus(param.train_dictspace,param.batch_size,param.sequence_len,param.max_train_sets)
    corpus.sentenceTotoken(param.dict_size)
    INPUT_SHAPE = [param.batch_size]
    OUTPUT_SHAPE = [param.batch_size]
    VOCAB_SIZE = corpus.get_vocab_size()
    model = Model(INPUT_SHAPE,OUTPUT_SHAPE,'lstm',param.rnn_cell_layer,param.rnn_cell_size,param.batch_size,VOCAB_SIZE,VOCAB_SIZE,param.embedding_len)
    saver = tf.train.Saver()
    global_steps = tf.get_variable(name='global_steps', dtype=tf.int32,
                                            initializer=tf.zeros(shape=[1], dtype=tf.int32))

    with tf.Session() as sess:
        '''
        restore
        '''
        ckpt = tf.train.get_checkpoint_state(param.train_workspace)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            last_step = int(ckpt.model_checkpoint_path.split('-')[-1])
            logging.info('Session has loaded the Checkpoint: %s' % ckpt.model_checkpoint_path)
        else:
            last_step = -1
            sess.run(tf.global_variables_initializer())
            logging.info('No Checkpoint founded!')

        decoder_xs = corpus.get_decoder_inputs()
        for step in range(last_step+1,param.training_steps):
            xs,ys,weights = corpus.next_batch()

            feed_dict = {
                model.learning_rate:min(0.22,param.lr*np.power(param.lr_decay,step / param.record_intervals))
            }
            for i,sxs in enumerate(model.xs):
                feed_dict[sxs.name] = xs[i]
            for i,sys in enumerate(model.ys):
                feed_dict[sys.name] = ys[i]
            for i, sys in enumerate(model.decoder_xs):
                feed_dict[sys.name] = decoder_xs[i]
            for i, sys in enumerate(model.weights):
                feed_dict[sys.name] = weights[i]
            loss,_,output = sess.run([model.cost,model.train_op,model.output],feed_dict=feed_dict)
            if step % param.record_intervals == 0:
                logging.info(
                    'Learning Rate: %f' % ((param.lr * np.power(param.lr_decay, step / param.record_intervals))))
                # run test_Set bad CASE wrong structure
                # xs, ys = corpus.get_test_batch()
                # costs = []
                # outputs = []
                # for index in range(0,len(xs),param.batch_size):
                #     if index + param.batch_size >= len(xs):
                #         continue
                #     feed_dict = {
                #         model.learning_rate: param.lr
                #     }
                #     for i, sxs in enumerate(model.xs):
                #         feed_dict[sxs.name] = xs[i+index]
                #     for i, sys in enumerate(model.ys):
                #         feed_dict[sys.name] = ys[i+index]
                #     for i, sys in enumerate(model.decoder_xs):
                #         feed_dict[sys.name] = decoder_xs[i]
                #     loss, output,r_loss = sess.run([model.cost,model.output,model.loss], feed_dict=feed_dict)
                #     print r_loss
                #     costs.append(loss)
                #     outputs.extend(output)
                # print costs
                # loss = tf.reduce_sum(costs)
                print 'step %d ---- loss:%s' % (step,loss)
                ys = np.transpose(ys)
                for i in range(10):
                    line = ''
                    for ids in range(8):
                        id = np.argmax(output[ids][i])
                        line = line + corpus.id_dict[id]
                    line = line + '\t'
                    for id in ys[i]:
                        line = line + corpus.id_dict[id]
                    print line.encode('utf-8')
                save_path = saver.save(sess, param.train_workspace+'/save_net.ckpt',global_step=step)
                print("Save to path: ", save_path)
            global_steps = global_steps + 1

    print 'hello world'


if __name__ == '__main__':
    param = HParam()
    run(param)

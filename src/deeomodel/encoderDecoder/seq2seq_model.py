# coding:utf-8
import tensorflow as tf
import logging

class Model:
    def __init__(self,input_shape,output_shape,cell_type,cell_layer,cell_size,batch_size,input_vocab_size,output_vocab_size,embedding_size,seq_len,sampled_nums):
        '''
        最基本的Encoder-Decoder模型
        :param input_shape:  输入的形状
        :param output_shape: 输出的形状
        :param cell_type: RNN使用哪种cell，可选lstm和gru
        :param cell_layer: 堆叠多层rnn网络，指定堆叠层数
        :param cell_size: 每层RNN的Cell的数量
        :param batch_size: 每一个batch的长度
        :param input_vocab_size: 输入的字典的大小
        :param output_vocab_size: 输出的字典的大小
        :param embedding_size: 这里将会进行词嵌入，这里表明词嵌入的长度
        :param seq_len:
        :param sampled_nums:
        '''
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

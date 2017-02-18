#coding:utf-8
import os
class HParam:
    def __init__(self):
        # File path for working
        print os.getcwd()
        self.test_workspace =  os.path.abspath('default_workspace')
        self.test_dictspace =  os.path.abspath('default_dictspace')
        self.train_dictspace = os.path.abspath('default_dictspace')
        self.train_workspace = os.path.abspath('default_workspace')

        # General (Constant)
        self.sequence_len = 10
        self.dict_size = 10500 # max number of tokens
        self.embedding_len = 50
        self.rnn_cell = 'lstm'
        self.rnn_cell_layer = 4
        self.rnn_cell_size = 256

        # Training
        self.max_train_sets = 100000
        self.batch_size = 100
        self.training_steps = 9000000
        self.record_intervals = 200
        self.lr = 0.4
        self.lr_decay = 0.99


        # Encoder (constant)
        self.encoder_dict_size = -1

        # Decoder (constant)
        self.decoder_dict_size = -1
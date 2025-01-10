import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils import *


class GraphConvolution():
    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def __call__(self, inputs):
        with tf.compat.v1.name_scope(self.name):
            x = inputs
            x = tf.compat.v1.nn.dropout(x, 1-self.dropout)
            x = tf.compat.v1.matmul(x, self.vars['weights'])
            x = tf.compat.v1.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs

class GraphConvolutionSparse():
    def __init__(self, input_dim, output_dim, adj, embeddings_nonzero, name, dropout=0., act=tf.compat.v1.nn.relu):
        self.name = name
        self.vars = {}
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.embeddings_nonzero = embeddings_nonzero

    def __call__(self, inputs):
        with tf.compat.v1.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1-self.dropout, self.embeddings_nonzero)
            x = tf.compat.v1.sparse_tensor_dense_matmul(x, self.vars['weights'])
            x = tf.compat.v1.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs

class InnerProductDecoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_r, num_groups, dropout=0, act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.num_r = num_r
        self.num_groups = num_groups
        self.dropout = dropout
        self.act = act
        group_size = input_dim // num_groups
        self.weight_shape = (group_size, group_size)

    # def build(self, input_shape):
        with tf.name_scope('inner_product_decoder_vars'):
            self.weight_matrix = self.add_weight(name='weight_matrix', shape=self.weight_shape,
                                                 initializer='glorot_uniform', trainable=True)

            self.attention_weights = self.add_weight(name='attention_weights', shape=(self.num_groups,1),
                                                     initializer='glorot_uniform', trainable=True)

            self.layer1 = tf.keras.layers.Dense(units=16, activation=tf.nn.relu)
            self.layer2 = tf.keras.layers.Dense(units=32, activation=tf.nn.relu)
            self.layer3 = tf.keras.layers.Dense(units=group_size, activation=tf.nn.relu)

    def call(self, inputs):
        with tf.name_scope('inner_product_decoder'):
            R = inputs[0:self.num_r, :]
            D = inputs[self.num_r:, :]
            R_groups = tf.split(R, self.num_groups, axis=1)
            D_groups = tf.split(D, self.num_groups, axis=1)
            similarity_scores_list = tf.TensorArray(dtype=tf.float32, size=self.num_groups)
            for i in range(self.num_groups):
                R_group = R_groups[i]
                D_group = D_groups[i]
                R_temp = self.layer1(R_group)
                R_temp = self.layer2(R_temp)
                R_branch_2 = self.layer3(R_temp)
                R_group = R_branch_2
                D_group = tf.transpose(D_group)
                x = tf.matmul(R_group, D_group)
                x = tf.reshape(x, [-1])
                outputs = self.act(x)
                similarity_scores_list = similarity_scores_list.write(i, outputs)
            output_matrix = similarity_scores_list.stack()
            output_matrix = tf.transpose(output_matrix)
            attention_scores = tf.nn.softmax(self.attention_weights)
            output = tf.matmul(output_matrix, attention_scores)
            output = tf.squeeze(output)
            return output
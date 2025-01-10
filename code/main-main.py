import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.reset_default_graph()
import gc
import random
from clac_metric import cv_model_evaluate
from utils import *
from model import GCNModel
from opt import Optimizer



def PredictScore( train_RNA_dis_matrix, RNA_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp):
    np.random.seed(seed)
    tf.compat.v1.reset_default_graph()
    tf.random.set_random_seed(seed)
    adj = constructHNet(train_RNA_dis_matrix, RNA_matrix, dis_matrix)
    adj = sp.csr_matrix(adj)
    association_nam = train_RNA_dis_matrix.sum()
    X = constructNet(train_RNA_dis_matrix)
    embeddings = sparse_to_tuple(sp.csr_matrix(X))
    num_embeddings = embeddings[2][1]
    embeddings_nonzero = embeddings[1].shape[0]
    adj_orig = train_RNA_dis_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))

    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'embeddings': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
        'adjdp': tf.compat.v1.placeholder_with_default(0., shape=())
    }
    model = GCNModel( num_groups, placeholders, num_embeddings, emb_dim,
                     embeddings_nonzero, adj_nonzero, train_RNA_dis_matrix.shape[0], name='GROUP')
    with tf.name_scope('optimizer'):

        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr, num_u=train_RNA_dis_matrix.shape[0], num_v=train_RNA_dis_matrix.shape[1], association_nam=association_nam)
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['embeddings']: embeddings})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        """
        feed_dict.update
        """
        if epoch % 10 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost))
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)

    feature_hidden = sess.run(model.embeddings, feed_dict=feed_dict)

    sess.close()
    # np.savetxt('../data/fea.txt', feature_hidden, fmt='%.10f')
    return res, feature_hidden

def cross_validation_experiment( RNA_dis_matrix, RNA_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp):
    index_matrix = np.mat(np.where(RNA_dis_matrix == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                 k_folds]).reshape(k_folds, CV_size,  -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
        random_index[association_nam - association_nam % k_folds:]
    random_index = temp
    metric = np.zeros((1, 7))
    print("seed=%d, evaluating piRNA-disease...." % (seed))
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k+1))
        train_matrix = np.matrix(RNA_dis_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0
        RNA_len = RNA_dis_matrix.shape[0]
        dis_len = RNA_dis_matrix.shape[1]
        RNA_disease_res, feature_hidden = PredictScore(
            train_matrix, RNA_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp)
        np.savetxt('../data/score.txt',RNA_disease_res,fmt='%.10f')
        print(RNA_disease_res)
        predict_y_proba = RNA_disease_res.reshape(RNA_len, dis_len)
        metric_tmp = cv_model_evaluate(
            RNA_dis_matrix, predict_y_proba, train_matrix)
        print(metric_tmp)
        metric += metric_tmp
        del train_matrix
        gc.collect()
    print(metric / k_folds)
    metric = np.array(metric / k_folds)
    return metric

if __name__ == "__main__":
    RNA_sim = np.loadtxt('../data/SimRNA.csv', delimiter=',',encoding='utf-8-sig')
    dis_sim = np.loadtxt('../data/SimDisease.csv', delimiter=',',encoding='utf-8-sig')
    RNA_dis_matrix = np.loadtxt('../data/piRNA-disease.csv', delimiter=',',encoding='utf-8-sig')

    # log = open('4_100_32_adjdp_dp.txt', mode='a', encoding='utf-8')  # 日志文件
    # print("------------------------------------------------------------", file=log)

    # lambad2 = [2, 4, 8, 16, 32, 64,128]
    # for num_groups in lambad2:
    num_groups = 32
    epoch = 1
    emb_dim = 256
    lr = 0.01
    adjdp = 0.8
    dp = 0.4
    simw = 6
    # print("adjdp= ", adjdp, file=log)
    # print("num_groups= ", num_groups)

    result = np.zeros((1, 7), float)
    average_result = np.zeros((1, 7), float)
    circle_time = 1
    for i in range(circle_time):
        result += cross_validation_experiment(
            RNA_dis_matrix, RNA_sim * simw, dis_sim * simw, i, epoch, emb_dim, dp, lr, adjdp)
    average_result = result / circle_time
    print(average_result)





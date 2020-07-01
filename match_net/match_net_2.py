import tensorflow as tf
import numpy as np
from sklearn.preprocessing import normalize
from match_net.cnn import CnnTextEncoder

np.random.seed(2)


class MatchingNet(tf.keras.Model):
    def __init__(self, pre_trained_embedding=None, cos_distance=True, retrain=False, linear_layer=0, filter_size=200, y_dim=0):
        super(MatchingNet, self).__init__()
        self.cos_distance = cos_distance
        self.encoder = CnnTextEncoder(pretrained_embedding=pre_trained_embedding, retrain=retrain, filters=filter_size, linear_layer=linear_layer)
        if y_dim > 0:
            self.linear_layer = tf.keras.layers.Dense(y_dim, activation="softmax")
    @tf.function
    def get_hidden(self, encoded_token_idxes):
        return self.encoder(encoded_token_idxes)

    @tf.function
    def embed_avg_supports(self, support_set_list):
        embed = list()
        for data_batch_by_label in support_set_list:
            cnn_rep = self.get_hidden(data_batch_by_label)
            cnn_rep = tf.reduce_mean(cnn_rep, axis=0)
            embed.append(cnn_rep)
        embed = tf.stack(embed, axis=0)
        return embed

    @tf.function
    def call(self, batch_train_index, support_set_embeddings, training=None, mask=None):
        batch_train_embeddings = self.get_hidden(batch_train_index)
        # support_set_embeddings = tf.cond(self.cos_distance == tf.constant(True, dtype=tf.bool), lambda: tf.nn.l2_normalize(support_set_embeddings, axis=1), lambda: support_set_embeddings)
        # batch_train_embeddings = tf.cond(self.cos_distance == tf.constant(True, dtype=tf.bool), lambda: tf.nn.l2_normalize(support_set_embeddings, axis=1),
        #                                  lambda:  batch_train_embeddings)

        if self.cos_distance:
            support_set_embeddings = tf.nn.l2_normalize(support_set_embeddings, axis=1)
            batch_train_embeddings = tf.nn.l2_normalize(batch_train_embeddings, axis=1)
        # reshape support set
        embed_dim = tf.shape(support_set_embeddings)[1]
        support_set_embeddings = tf.expand_dims(support_set_embeddings, axis=0)
        support_set_embeddings = tf.reshape(
            support_set_embeddings, shape=(1, embed_dim, -1)
        )

        # reshape target set
        embed_dim = tf.shape(batch_train_embeddings)[1]

        batch_target_embeddings = tf.expand_dims(batch_train_embeddings, axis=2)
        batch_target_embeddings = tf.reshape(
            batch_target_embeddings, shape=(-1, embed_dim, 1)
        )

        logits = tf.reduce_sum(
            batch_target_embeddings * support_set_embeddings, axis=1
        )  # batchsize * label size


        return logits

    @tf.function
    def constrastive_learn(self, support_set_ids, support_set_ids2, training=None, mask=None):

        support_set_embeddings = self.get_hidden(support_set_ids)
        support_set_embeddings2 = self.get_hidden(support_set_ids2)
        similarity_matrix = tf.matmul(support_set_embeddings, tf.transpose(support_set_embeddings2))
        positive = tf.linalg.diag_part(similarity_matrix)
        negative = similarity_matrix - positive
        return positive, negative


    @tf.function
    def classify(self, X):
        result = self.encoder(X)
        return self.linear_layer(result)
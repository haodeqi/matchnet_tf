import tensorflow as tf
import numpy as np
from sklearn.preprocessing import normalize
from match_net.cnn import CnnTextEncoder

np.random.seed(2)


class MatchingNet(tf.keras.Model):
    def __init__(self, pre_trained_embedding=None, cos_distance = True):
        super(MatchingNet, self).__init__()
        self.cos_distance = cos_distance
        self.encoder = CnnTextEncoder(pretrained_embedding=pre_trained_embedding)

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
    def call(
        self,
        batch_train_index,
        support_set_embeddings,
        training=None,
        mask=None,
    ):
        batch_train_embeddings = self.get_hidden(batch_train_index)
        if self.cos_distance:
            support_set_embeddings = tf.nn.l2_normalize(support_set_embeddings, axis=1)
            batch_train_embeddings = tf.nn.l2_normalize(batch_train_embeddings, axis=1)

        # reshape support set
        label_dim, embed_dim = support_set_embeddings.shape
        support_set_embeddings = tf.expand_dims(support_set_embeddings, axis=0)
        support_set_embeddings = tf.reshape(support_set_embeddings, shape=(1, embed_dim, label_dim))

        # reshape target set
        batch_size, embed_dim = batch_train_embeddings.shape

        batch_target_embeddings = tf.expand_dims(batch_train_embeddings,axis =2)
        batch_target_embeddings = tf.reshape(batch_target_embeddings, shape=(batch_size, embed_dim, 1))

        logits = tf.reduce_sum(batch_target_embeddings*support_set_embeddings, axis = 1) # batchsize * label size

        return logits

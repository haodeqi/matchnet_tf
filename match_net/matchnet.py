import tensorflow as tf
import numpy as np
from sklearn.preprocessing import normalize

np.random.seed(2)


class FEC_G(tf.keras.layers.Layer):
    def __init__(self, hidden_size):
        super(FEC_G, self).__init__()
        self.bid_lstm = tf.keras.layers.Bidirectional(
            layer=tf.keras.layers.LSTM(
                units=hidden_size, return_sequences=True, time_major=False
            )
        )
        self.hidden_size = hidden_size

    def __call__(self, support_set_embeddings, training=None, mask=None):
        output = self.bid_lstm(support_set_embeddings)
        output = tf.concat(output, axis=0)  # [B, N, h*2]
        return (
            support_set_embeddings
            + output[:, :, : self.hidden_size]
            + output[:, :, self.hidden_size :]  # [B, N, h]
        )


class FEC_F(tf.keras.layers.Layer):
    def __init__(self, hidden_size, K=4):
        super(FEC_F, self).__init__()
        self.att_lstm = tf.keras.layers.LSTMCell(units=hidden_size)
        self.K = K

    def __call__(
        self,
        support_set_embeddings,  # [B, N, h]
        target_set_embeddings,
        training=None,
        mask=None,
    ):
        B, N, D = support_set_embeddings.shape.as_list()
        h = tf.zeros(shape=(B, D))
        c = tf.zeros(shape=(B, D))
        prev_states = [c, h]
        for i in range(self.K):
            output, state = self.att_lstm(
                target_set_embeddings, states=prev_states
            )  # output B,H, state[1]
            state[1] = state[1] + target_set_embeddings
            c_based_attention = tf.nn.softmax(
                tf.reduce_sum(
                    tf.expand_dims(state[1], axis=1) * support_set_embeddings, axis=2
                ),
                axis=1,
            )
            r = tf.reduce_sum(
                support_set_embeddings * tf.expand_dims(c_based_attention, axis=2),
                axis=1,
            )
            prev_states = (state[0], state[1] + r)
        return output  # [B, h]


class MatchingNet(tf.keras.Model):
    def __init__(self, x_dim, y_dim, context_encoding=False):
        super(MatchingNet, self).__init__()
        self.context_encoding = context_encoding
        if self.context_encoding:
            self.fec_f = FEC_F(x_dim)
            self.fec_g = FEC_G(x_dim)
        self.x_dim = x_dim
        self.y_dim = y_dim


    @tf.function
    def call(
        self,
        support_set_embeddings,
        support_set_y_lab,
        target_embeddings,
        n_way=3,
        training=None,
        mask=None,
    ):
        support_set_embeddings = tf.nn.l2_normalize(support_set_embeddings, axis=-1)
        target_embeddings = tf.nn.l2_normalize(target_embeddings, axis=-1)
        if self.context_encoding:
            target_embeddings = self.fec_f(support_set_embeddings, target_embeddings)
            support_set_embeddings = self.fec_g(support_set_embeddings)
        attention_weights = tf.nn.softmax(
            tf.reduce_sum(
                tf.expand_dims(target_embeddings, axis=1) * support_set_embeddings,
                axis=-1,
            ),
            axis=1,
        )
        logits = tf.reduce_sum(
            tf.expand_dims(attention_weights, axis=2)
            * tf.one_hot(support_set_y_lab, depth=n_way),
            axis=1,
        )
        return logits

    def predict(
        self,
        x_supp,
        y_supp,
        x_target,
        n_way=3,
        batch_size=None,
        verbose=0,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):
        results = list()
        for x_t in x_target:
            x_t = x_t[np.newaxis, :]
            logits = self.call(x_supp, y_supp, x_t, n_way)
            results.append(tf.argmax(logits, axis=1).numpy())
        return np.array(results).reshape(-1)

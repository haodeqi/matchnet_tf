from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D


class PreTrainedEmbeddding(tf.keras.layers.Layer):
    def __init__(self, embeddings, retrain=False, rate=0.1, **kwargs):
        """"Instantiate the layer using a pre-defined embedding matrix."""
        super().__init__(**kwargs)
        self.embeddings = tf.Variable(embeddings, trainable=retrain)
        # if you want to add some dropout (or normalization, etc.)
        self.dropout = tf.keras.layers.Dropout(rate=rate)

    def call(self, inputs, training=None):
        """Embed some input tokens and optionally apply dropout."""
        output = tf.nn.embedding_lookup(self.embeddings, inputs)
        return self.dropout(output, training=training)


class CnnTextEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        max_features=5000,
        maxlen=400,
        embedding_dims=200,
        filters=200,
        kernel_size=3,
        pretrained_embedding=None,
        retrain=False,
    ):
        super(CnnTextEncoder, self).__init__()
        self.model = Sequential()

        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        if pretrained_embedding is not None:
            self.model.add(
                PreTrainedEmbeddding(pretrained_embedding, trainable=retrain)
            )
        else:
            self.model.add(Embedding(max_features, embedding_dims, input_length=maxlen))

        self.model.add(
            Conv1D(filters, kernel_size, padding="valid", activation="relu", strides=1)
        )
        self.model.add(Activation("relu"))
        self.model.add(GlobalMaxPooling1D())

    @tf.function
    def call(self, X):
        return self.model(X)


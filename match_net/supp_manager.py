from match_net.data_set import DataSet
import tensorflow as tf
import numpy as np


class SuppManager:
    def __init__(self, data_set_dict: {str: DataSet}, model):
        self.data_set_dict = data_set_dict
        self.model = model

    def select_all_avg_support_set(self, task_id):
        data = self.data_set_dict[task_id]
        embed = list()
        for label in data.idx2label:
            data_batch_by_label = data.elabel2edata[label]
            cnn_rep = self.model.get_hidden(data_batch_by_label)
            cnn_rep = tf.reduce_mean(cnn_rep, axis=1)
            embed.append(cnn_rep)
        embed = tf.stack(embed, axis=0)
        return embed

    def sample_avg_support_set(self, task_id, sample_size=2):
        data = self.data_set_dict[task_id]
        embed = list()
        for label in data.idx2label:
            data_batch_by_label = data.elabel2edata[label]
            if data_batch_by_label.shape[0] > sample_size:
                idxes = np.arange(data_batch_by_label.shape[0]).tolist()
                sampled_idx = np.random.choice(idxes, size=sample_size, replace=False)
                data_batch_by_label = data_batch_by_label[sampled_idx]

            cnn_rep = self.model.get_hidden(data_batch_by_label)
            cnn_rep = tf.reduce_mean(cnn_rep, axis=0)
            embed.append(cnn_rep)
        embed = tf.stack(embed, axis=0)
        return embed

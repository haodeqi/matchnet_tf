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

    def sample_constrastive_set_id(self, task_id):
        data = self.data_set_dict[task_id]
        id1 = list()
        id2 = list()
        for label in data.idx2label:
            data_batch_by_label = data.elabel2edata[label]
            # sample size means how much sample per label.
            # for _ in range(min(data_batch_by_label.shape[0], sample_size)):
            idxes = np.arange(data_batch_by_label.shape[0]).tolist()
            if len(idxes) > 1:

                sampled_idx = np.random.choice(idxes, size=2, replace=False) # the size here is always 2. 1 for each list
                #data_batch_by_label = data_batch_by_label[sampled_idx]
                # sample
                id1.append(data_batch_by_label[sampled_idx[0]].reshape(-1))
                id2.append(data_batch_by_label[sampled_idx[1]].reshape(-1))
            else:
                id1.append(data_batch_by_label[idxes].reshape(-1))
                id2.append(data_batch_by_label[idxes].reshape(-1))
        return id1, id2
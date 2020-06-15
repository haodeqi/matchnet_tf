import os
from match_net.data_set import DataSet
from collections import defaultdict
import numpy as np
from match_net.cnn import CnnTextEncoder
from match_net.supp_manager import SuppManager
from match_net.match_net_2 import MatchingNet
from tqdm import tqdm, trange
from logging import getLogger
import argparse
import tensorflow as tf
tf.executing_eagerly()

LOGGER = getLogger(__file__)

input_path = "/Users/haode/tasks/auto-ml/large_scale_test_data_2019_0716_en"
data_list_path = "/Users/haode/git/matchnet/match_net/workspace_list"
embedding_path = "/Users/haode/git/DiverseFewShot_Amazon/few_shot_code/glove.6B/glove.6B.200d.txt"
specials = ["<unk>", "<pad>"]
epochs = 100
if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description="run training for large scale test")
    parser.add_argument("--input_path",help="path to the input folder")
    parser.add_argument("--data_list", help="path to the ids of datasets")
    parser.add_argument("--embedding_path", help = "path to the embedding file")


    def get_glove(embedding_path, specials = ["<unk>", "<pad>"]):
        embedding_matrix = list()
        token2id = defaultdict(int)
        with open(embedding_path,"r", encoding="utf-8") as file:

            for line in tqdm(file):
                line = line.strip().split(" ")
                token2id[line[0].lower()]
                embedding_matrix.append(np.array(line[1:], dtype=np.float))
            embed_dim = embedding_matrix[-1].shape[0]
            for special in specials:
                token2id[special]
                embedding_matrix.append(np.zeros(shape=(embed_dim), dtype=np.float))
        return np.array(embedding_matrix), token2id


    def get_data_list(data_list_path):
        data_list = list()
        with open(data_list_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip().split()
                data_list.append(line[0])
        return data_list
    LOGGER.info("start loading embedings")
    data_list = get_data_list(data_list_path)
    embed_matrix, token2id = get_glove(embedding_path)
    LOGGER.info("finish loading embedings")

    LOGGER.info("start loading data")
    train_data_sets = dict()
    dev_data_sets = dict()
    test_data_sets = dict()
    LOGGER.info("finish loading data")
    for data_id in tqdm(data_list):
        train_data_sets[data_id] = DataSet(os.path.join(input_path,data_id+".train"),token_map=token2id,sequence_len=100)
        label2idx = train_data_sets[data_id].label2idx
        dev_data_sets[data_id] = DataSet(os.path.join(input_path, data_id + ".dev"), token_map=token2id, label2idx=label2idx,sequence_len=100)

    # cnn_encoder = CnnTextEncoder(maxlen = 100,
    #                             filters = 200,
    #                             kernel_size = 3,
    #                             pretrained_embedding=embed_matrix)
    match_net = MatchingNet(pre_trained_embedding=embed_matrix, cos_distance=True)
    supp_manager = SuppManager(model=match_net, data_set_dict=train_data_sets)
    # print information about the data
    num_batch_total = 0
    batch_size = 10
    iter_per_sample = 2
    for data_set_id in tqdm(train_data_sets):
        train_size = train_data_sets[data_set_id].encoded_data.shape[0]
        num_batch_total += int(np.ceil(train_size / batch_size))

    iterations = np.ceil(num_batch_total*epochs/iter_per_sample).astype(np.int)

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-1,
    #     decay_steps=100,
    #     decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(lr=0.01, clipvalue=.5)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


    @tf.function
    def train_step(train_set, supp_embed, labels, model, loss_object):
        with tf.GradientTape() as tape:
            predictions = model(train_set, supp_embed)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)


    task_ids = list(train_data_sets.keys())
    for i in trange(iterations):
        task_id = np.random.choice(task_ids)
        train_iter, dev_iter = train_data_sets[task_id], dev_data_sets[task_id]
        j = 0

        for train_batch, label_batch in train_iter.generate(batch_size=batch_size):
            if j > iter_per_sample:
                train_iter.shuffle()
                break
            supp_embed = supp_manager.sample_avg_support_set(task_id, sample_size=10)
            #embeded_train = match_net.get_hidden(train_batch)
            train_step(train_batch, supp_embed, label_batch, match_net, loss_object)
            j += 1
        template = 'Epoch {}, Loss: {}, Accuracy: {}'
        print(template.format(i + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100))

        if i % 15 == 0:
            correct = 0
            total = 0
            embedded_support_set = supp_manager.sample_avg_support_set(task_id, sample_size=float("inf"))
            for dev_batch, label_batch in dev_iter.generate(batch_size=batch_size):
                predictions = match_net(dev_batch, embedded_support_set)
                correct += np.sum(tf.argmax(predictions, axis=1) == label_batch)
                total += len(label_batch)
            print("accuracy is {} for task {}".format(correct/total, task_id))
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

SPECIALS = ["<unk>", "<pad>"]


def argument_parsing():
    parser = argparse.ArgumentParser(description="run training for large scale test")
    parser.add_argument("--input_path", help="path to the input folder")
    parser.add_argument("--data_list", help="path to the ids of datasets")
    parser.add_argument("--embedding_path", help="path to the embedding file")
    parser.add_argument("--batch_size", help="size of training", type=int, default=24)
    parser.add_argument(
        "--epochs", help="number of total passage to dataset", type=int, default=10
    )
    parser.add_argument(
        "--lr", help="the size of learning rate", type=float, default=0.01
    )
    parser.set_defaults(cosine=False)
    parser.add_argument(
        "--cosine", action="store_true", help="use cosine distance", dest="cosine"
    )
    parser.set_defaults(retrain=False)
    parser.add_argument(
        "--retrain", action="store_true", help="retrain embeddings", dest="retrain"
    )
    parser.add_argument(
        "--output_dir", help="where to store the output", type=str, default="./"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parsing()
    input_path = args.input_path
    data_list_path = args.data_list
    embedding_path = args.embedding_path
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    cosine = args.cosine
    retrain = args.retrain
    output_dir = args.output_dir

    def get_glove(embedding_path, specials=("<unk>", "<pad>")):
        embedding_matrix = list()
        token2id = dict()
        with open(embedding_path, "r", encoding="utf-8") as file:
            i = 0
            for line in tqdm(file):
                line = line.strip().split(" ")
                token2id[line[0].lower()] = i
                i += 1
                embedding_matrix.append(np.array(line[1:], dtype=np.float))
            embed_dim = embedding_matrix[-1].shape[0]
            for special in specials:
                token2id[special] = i
                i += 1
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
    embed_matrix, token2id = get_glove(embedding_path, specials=SPECIALS)
    LOGGER.info("finish loading embedings")
    LOGGER.info("start loading data")
    train_data_sets = dict()
    dev_data_sets = dict()
    test_data_sets = dict()
    LOGGER.info("finish loading data")
    for data_id in tqdm(data_list):
        train_data_sets[data_id] = DataSet(
            os.path.join(input_path, data_id + ".train"),
            token_map=token2id,
            sequence_len=100,
        )
        label2idx = train_data_sets[data_id].label2idx
        dev_data_sets[data_id] = DataSet(
            os.path.join(input_path, data_id + ".dev"),
            token_map=token2id,
            label2idx=label2idx,
            sequence_len=100,
        )

    match_net = MatchingNet(pre_trained_embedding=embed_matrix, cos_distance=cosine, retrain=retrain)
    supp_manager = SuppManager(model=match_net, data_set_dict=train_data_sets)
    # print information about the data
    num_batch_total = 0
    iter_per_sample = 2
    for data_set_id in tqdm(train_data_sets):
        train_size = train_data_sets[data_set_id].encoded_data.shape[0]
        num_batch_total += int(np.ceil(train_size / batch_size))

    iterations = np.ceil(num_batch_total * epochs / iter_per_sample).astype(np.int)

    optimizer = tf.keras.optimizers.Adam(lr=lr, clipnorm=0.5)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    @tf.function
    def train_step(train_set, supp_embed, labels, model, loss_object):
        with tf.GradientTape() as tape:
            predictions = model(train_set, supp_embed)
            tf.print(labels)
            tf.print(tf.argmax(predictions, axis=1))
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
            train_step(train_batch, supp_embed, label_batch, match_net, loss_object)
            j += 1
        template = "Epoch {}, Loss: {}, Accuracy: {}"
        print(
            template.format(i + 1, train_loss.result(), train_accuracy.result() * 100)
        )

        if i % 15 == 0:
            correct = 0
            total = 0
            embedded_support_set = supp_manager.sample_avg_support_set(
                task_id, sample_size=float("inf")
            )
            for dev_batch, label_batch in dev_iter.generate(batch_size=batch_size):
                predictions = match_net(dev_batch, embedded_support_set)
                correct += np.sum(tf.argmax(predictions, axis=1) == label_batch)
                total += len(label_batch)
            print("accuracy is {} for task {}".format(correct / total, task_id))

    results = list()
    for task_id in task_ids:
        dev_iter = dev_data_sets[task_id]
        embedded_support_set = supp_manager.sample_avg_support_set(
            task_id, sample_size=float("inf")
        )
        correct = 0
        total = 0
        for dev_batch, label_batch in dev_iter.generate(batch_size=batch_size):
            predictions = match_net(dev_batch, embedded_support_set)
            correct += np.sum(tf.argmax(predictions, axis=1) == label_batch)
            total += len(label_batch)
        results.append((task_id, correct/total))

    with open(os.path.join(output_dir, "statistics.txt"), "w", encoding="utf-8") as file:
        file.writelines([result[0] + "\t" + str(result[1]) + "\n" for result in results])
    # tf.saved_model.save(match_net, os.path.join(output_dir, "model_files"))

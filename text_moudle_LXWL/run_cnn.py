#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab
import pandas as pd
from function import Index

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

base_dir = 'data/name2category'
train_dir = os.path.join(base_dir, 'name2category.train.txt')
test_dir = os.path.join(base_dir, 'name2category.test.txt')
val_dir = os.path.join(base_dir, 'name2category.val.txt')
vocab_dir = os.path.join(base_dir, 'name2category.vocab.txt')
pred_dir = os.path.join(base_dir, 'name2category.predict.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)
            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            feed_dict[model.keep_prob] = config.dropout_keep_prob
            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

class name2subcategory():

    def __init__(self):
        print('Configuring CNN model...')
        if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
            build_vocab(train_dir, vocab_dir, config.vocab_size)
        self.categories, cat_to_id = read_category()
        words, self.word_to_id = read_vocab(vocab_dir)
        self.table = pd.read_excel('predict_check_data.xls')
        category_set = list(set(self.table['name'].tolist()))
        self.config = TCNNConfig(len(list(category_set)))
        self.config.vocab_size = len(words)
        self.model = TextCNN(self.config)
        self.categories = list(set(self.table['name'].tolist()))
        self.categories.sort(key = self.table['name'].tolist().index)

    def namelyst_predict(self, contents):
        import keras as kr

        #print("Loading predicted data...")
        data_id = []
        for i in range(len(contents)):
            data_id.append([self.word_to_id[x] for x in contents[i] if x in self.word_to_id])
        x_pred = kr.preprocessing.sequence.pad_sequences(data_id, self.config.seq_length)
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=save_path)  # 读取保存的模型
        batch_size = 128
        data_len = len(x_pred)
        num_batch = int((data_len - 1) / batch_size) + 1
        y_pred_cls = np.zeros(shape=len(x_pred), dtype=np.int32)  # 保存预测结果
        for i in range(num_batch):  # 逐批次处理
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            feed_dict = {
                self.model.input_x: x_pred[start_id:end_id],
                self.model.keep_prob: 1.0
            }
            y_pred_cls[start_id:end_id] = session.run(self.model.y_pred_cls, feed_dict=feed_dict)    # y_pred_cls为预测的list。
        y_pred_list = []
        for m in range(len(y_pred_cls)):
            y_pred_list.append(self.categories[y_pred_cls[m]])

        return y_pred_list


def predict():
    import keras as kr

    table = pd.read_excel('predict_check_data.xls')
    categories = list(set(table['name'].tolist()))
    categories.sort(key = table['name'].tolist().index)
    print("Loading predicted data...")
    f = open(pred_dir, 'r', encoding='utf-8', errors='ignore')
    contents = []
    for line in f:
        try:
            if line:
                contents.append(list(line))
        except:
            pass
    f.close()

    data_id = []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
    x_pred = kr.preprocessing.sequence.pad_sequences(data_id, config.seq_length)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型
    batch_size = 128
    data_len = len(x_pred)
    num_batch = int((data_len - 1) / batch_size) + 1
    y_pred_cls = np.zeros(shape=len(x_pred), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_pred[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)    # y_pred_cls为预测的list。
    y_pred_list = []
    for m in range(len(y_pred_cls)):
        y_pred_list.append(categories[y_pred_cls[m]])
    f = open('predicted_data.txt','a+', encoding='utf-8', errors='ignore')
    for n in range(len(contents)):
        f.write(y_pred_list[n] + "\t") 
        f.write(''.join(contents[n]))
    f.close()
    return 0
    



if __name__ == '__main__':
    
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test', 'predict']:
        raise ValueError("""usage: python run_cnn.py [train / test / predict]""")

    print('Configuring CNN model...')
    table = pd.read_excel('predict_check_data.xls')
    SubCategoryName_list = table['name'].tolist()
    category_set = list(set(SubCategoryName_list))
    category_set.sort(key = SubCategoryName_list.index)
    config = TCNNConfig(len(category_set))
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    #print(config.vocab_size)
    model = TextCNN(config)
    
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'predict':
        predict()
    else:
        test()

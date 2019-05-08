import tensorflow as tf
from model import Mdoel
from dataset import DataSet
from common import Config
import os


def train():
    m = Mdoel()
    data_train = DataSet('train')
    data_test = DataSet('test')

    def get_inputs():
        """
        模型输入tensor
        """
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
        target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
        max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
        source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

        return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length

    # 构造graph
    train_graph = tf.Graph()
    valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths = next(data_test.one_epoch_generator())
    source_letter_to_int = data_train.chinese_2_index_dict
    target_letter_to_int = data_train.english_2_index_dict

    with train_graph.as_default():
        # 获得模型输入
        input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()

        training_decoder_output, predicting_decoder_output = m.seq2seq_model(input_data, targets,
                                                                             target_sequence_length,
                                                                             max_target_sequence_length,
                                                                             source_sequence_length,
                                                                             source_vocab_size=len(source_letter_to_int),
                                                                             target_letter_to_int=target_letter_to_int)

        training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')

        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            # Loss function
            cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)

        # Set tf summary
        tboard_save_dir = Config.tboard_save_dir
        os.makedirs(tboard_save_dir, exist_ok=True)
        tf.summary.scalar(name='train_loss', tensor=cost)
        merged = tf.summary.merge_all()

        # Set sess configuration
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(tboard_save_dir)
        summary_writer.add_graph(sess.graph)

    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        print('Initialiation finished!')

        for epoch_i in range(1, Config.epochs + 1):
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                    data_train.one_epoch_generator()):
                _, loss, summary = sess.run(
                    [train_op, cost, merged],
                    feed_dict={input_data: sources_batch,
                     targets: targets_batch,
                     lr: Config.learning_rate,
                     target_sequence_length: targets_lengths,
                     source_sequence_length: sources_lengths})
                summary_writer.add_summary(summary, epoch_i)
                if batch_i % Config.display_step == 0:
                    # 计算validation loss
                    validation_loss = sess.run(
                        [cost],
                        {input_data: valid_sources_batch,
                         targets: valid_targets_batch,
                         lr: Config.learning_rate,
                         target_sequence_length: valid_targets_lengths,
                         source_sequence_length: valid_sources_lengths})

                    print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                          .format(epoch_i,
                                  Config.epochs,
                                  batch_i,
                                  Config.train_size // Config.batch_size,
                                  loss,
                                  validation_loss[0]))

        # 保存模型
        saver = tf.train.Saver()
        saver.save(sess, Config.model_save_dir)
        print('Model Trained and Saved')


if __name__ == '__main__':
    train()

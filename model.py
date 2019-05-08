import tensorflow as tf
from common import Config
from tensorflow.python.layers.core import Dense


class Mdoel:
    def __init__(self):
        self.encoding_embedding_size = Config.encoder_embeding_size
        self.decoding_embedding_size = Config.decoder_embeding_size
        self.batch_size = Config.batch_size

    def get_encoder_layer(self, input_data, source_sequence_length, source_vocab_size):
        """
        构造Encoder层

        - input_data: 输入tensor
        - source_sequence_length: 源数据的序列长度
        - source_vocab_size: 源数据的词典大小
        """
        # Encoder embedding
        encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, self.encoding_embedding_size)

        # RNN cell
        def get_lstm_cell(rnn_size):
            lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(Config.encoder_rnn_size) for _ in range(Config.num_layers)])

        encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input,
                                                          sequence_length=source_sequence_length, dtype=tf.float32)

        return encoder_output, encoder_state

    def decoding_layer(self, target_letter_to_int,
                       target_sequence_length, max_target_sequence_length, encoder_state, decoder_input):
        """
        构造Decoder层

        - target_letter_to_int: target数据的映射表
        - decoding_embedding_size: embed向量大小
        - target_sequence_length: target数据序列长度
        - max_target_sequence_length: target数据序列最大长度
        - encoder_state: encoder端编码的状态向量
        - decoder_input: decoder端输入
        """
        # 1. Embedding
        target_vocab_size = len(target_letter_to_int)
        decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, self.decoding_embedding_size]))
        decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

        # 2. 构造Decoder中的RNN单元
        def get_decoder_cell(rnn_size):
            decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                                   initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return decoder_cell

        cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(Config.decoder_rnn_size) for _ in range(Config.num_layers)])

        # 3. Output全连接层
        output_layer = Dense(target_vocab_size,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        # 4. Training decoder
        with tf.variable_scope("decode"):
            # 得到help对象
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                                sequence_length=target_sequence_length,
                                                                time_major=False)
            # 构造decoder
            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                               training_helper,
                                                               encoder_state,
                                                               output_layer)
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                           impute_finished=True,
                                                                           maximum_iterations=max_target_sequence_length)
            # 5. Predicting decoder
            # 与training共享参数
        with tf.variable_scope("decode", reuse=True):
            # 创建一个常量tensor并复制为batch_size的大小
            start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']], dtype=tf.int32), [self.batch_size],
                                   name='start_tokens')
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                         start_tokens,
                                                                         target_letter_to_int['<EOS>'])
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                                 predicting_helper,
                                                                 encoder_state,
                                                                 output_layer)
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder, impute_finished=True,
                                                                    maximum_iterations=max_target_sequence_length)

        return training_decoder_output, predicting_decoder_output

    @staticmethod
    def process_decoder_input(data, vocab_to_int, batch_size):
        """
        补充<GO>，并移除最后一个字符
        """
        # cut掉最后一个字符
        ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

        return decoder_input

    def seq2seq_model(self, input_data, targets, target_sequence_length,
                      max_target_sequence_length, source_sequence_length,
                      source_vocab_size, target_letter_to_int):
        # 获取encoder的状态输出
        _, encoder_state = self.get_encoder_layer(input_data, source_sequence_length, source_vocab_size)

        # 预处理后的decoder输入
        decoder_input = self.process_decoder_input(targets, target_letter_to_int, Config.batch_size)

        # 将状态向量与输入传递给decoder
        training_decoder_output, predicting_decoder_output = self.decoding_layer(target_letter_to_int,
                                                                            target_sequence_length,
                                                                            max_target_sequence_length,
                                                                            encoder_state,
                                                                            decoder_input)

        return training_decoder_output, predicting_decoder_output

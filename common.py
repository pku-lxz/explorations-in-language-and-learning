import os


class Config:
    data_dir = '/Users/lamprad/Desktop/cmn-eng/'

    model_save_dir = os.path.join(data_dir, 'tf_ckpt')
    tboard_save_dir = os.path.join(data_dir, 'tf_log')
    train_size = 18000
    test_size = 21007 - train_size
    encoder_embeding_size = 20
    decoder_embeding_size = 20
    batch_size = 64
    learning_rate = 0.01

    # RNN Size
    encoder_rnn_size = 50
    decoder_rnn_size = 50

    # Number of Layers
    num_layers = 28

    display_step = 50  # 每隔50轮输出loss
    epochs = 20

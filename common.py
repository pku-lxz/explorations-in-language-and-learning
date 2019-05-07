import os


class config:
    data_dir = '/Users/lamprad/Desktop/cmn-eng/'
    model_save_dir = os.path.join(data_dir, 'tf_ckpt')
    tboard_save_dir = os.path.join(data_dir, 'tf_log')
    train_size = 18000
    test_size = 21007 - train_size
    input_size = 50
    train_batch_size = 64
    test_batch_size = 64

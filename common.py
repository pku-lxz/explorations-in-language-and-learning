import os


class config:
    data_dir = '/Users/lamprad/Desktop/cmn-eng/'
    model_save_dir = os.path.join(data_dir, 'tf_ckpt')
    tboard_save_dir = os.path.join(data_dir, 'tf_log')
    train_size = 100
    test_size = 70

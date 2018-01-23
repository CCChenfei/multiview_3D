"""
TRAIN LAUNCHER

"""

import configparser
from hourglassModel import HourglassModel
from datagen import DataGenerator
import tensorflow as tf
import scipy.io

def process_config(conf_file):
    """
    """
    params = {}
    config = configparser.ConfigParser()
    config.read(conf_file)
    for section in config.sections():
        if section == 'DataSetHG':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Network':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Train':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Validation':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Saver':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
    return params


if __name__ == '__main__':
    print('--Parsing Config File')
    params = process_config('config.cfg')
    print('--Creating Dataset')
    dataset1 = DataGenerator(params['joint_list'], params['img_directory1'], params['training_txt_file1'],remove_joints=params['remove_joints'], img_dir_test=params['img_directory_test1'], test_data_file=params['test_txt_file1'],train_3D_gt=params['train_3d_gt'],test_3D_gt=params['test_3d_gt'])
    dataset2 = DataGenerator(params['joint_list'], params['img_directory2'], params['training_txt_file2'],remove_joints=params['remove_joints'], img_dir_test=params['img_directory_test2'], test_data_file=params['test_txt_file2'])
    dataset3 = DataGenerator(params['joint_list'], params['img_directory3'], params['training_txt_file3'],remove_joints=params['remove_joints'], img_dir_test=params['img_directory_test3'], test_data_file=params['test_txt_file3'])
    dataset4 = DataGenerator(params['joint_list'], params['img_directory4'], params['training_txt_file4'],remove_joints=params['remove_joints'], img_dir_test=params['img_directory_test4'], test_data_file=params['test_txt_file4'])
    dataset1._create_train_table()
    dataset1._randomize()
    dataset1._create_sets()
    dataset2._create_train_table()
    dataset2._randomize()
    dataset2._create_sets()
    dataset3._create_train_table()
    dataset3._randomize()
    dataset3._create_sets()
    dataset4._create_train_table()
    dataset4._randomize()
    dataset4._create_sets()
    dataset = [dataset1, dataset2, dataset3, dataset4]

    camera1 = scipy.io.loadmat(params['camera1'])
    camera2 = scipy.io.loadmat(params['camera2'])
    camera3 = scipy.io.loadmat(params['camera3'])
    camera4 = scipy.io.loadmat(params['camera4'])
    cam = [camera1, camera2, camera3, camera4]

    model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'],nLow=params['nlow'], outputDim=params['num_joints'], batch_size=params['batch_size'],training=True, drop_rate=params['dropout_rate'],lear_rate=params['learning_rate'], decay=params['learning_rate_decay'],decay_step=params['decay_step'], dataset=dataset, name=params['name'],logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'],joints=params['joint_list'],cam=cam)
    model.generate_model()
    model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'],dataset=None,load=None)

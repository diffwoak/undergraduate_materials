import os
import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time
import shutil
import random

from trainer import VolumeClassifier
from data_utils.csv_reader import csv_reader_single
from config import INIT_TRAINER, SETUP_TRAINER, VERSION, CURRENT_FOLD, WEIGHT_PATH_LIST,FOLD_NUM
from converter.common_utils import save_as_hdf5

# 获取交叉验证所需的训练集和验证集
def get_cross_validation(path_list, fold_num, current_fold):
    # 按照fold_num平分数据集，根据current_fold得到当前fold数据集
    _len_ = len(path_list) // fold_num
    train_id = []
    validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        validation_id.extend(path_list[start_index:])
        train_id.extend(path_list[:start_index])
    else:
        validation_id.extend(path_list[start_index:end_index])
        train_id.extend(path_list[:start_index])
        train_id.extend(path_list[end_index:])

    print(f'Total sample number:{len(path_list)}, train sample number:{len(train_id)}, val sample number:{len(validation_id)}')
    return train_id, validation_id

# 查询神经网络模型的参数数量信息
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Net parameter number Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":
    ###############################################
    # 命令行参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--mode',
                        default='train',
                        choices=["train-cross","train", "inf","inf-cross"],
                        help='choose the mode',
                        type=str)
    parser.add_argument('-s',
                        '--save',
                        default='yes',
                        choices=['no', 'n', 'yes', 'y'],
                        help='save the forward middle features or not',
                        type=str)
    parser.add_argument('-n',
                        '--net_name',
                        #default=None,
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152","resnext50_32x4d","resnext101_32x8d","resnext101_64x4d","wide_resnet50_2","wide_resnet101_2",
           "vit_b_16","vit_b_32","vit_l_16","vit_l_32","vit_h_14"],
                        help='override the INIT_TRAINER[\'net_name\'] of config.py',
                        type=str)
    parser.add_argument('-l',
                        '--lr',
                        #default=None,
                        help='override the INIT_TRAINER[\'lr\'] of config.py',
                        type=float)
    parser.add_argument('-e',
                        '--n_epoch',
                        #default=None,
                        help='override the INIT_TRAINER[\'n_epoch\'] of config.py',
                        type=int)
    parser.add_argument('-c',
                        '--num_classes',
                        #default=None,
                        help='override the INIT_TRAINER[\'num_classes\'] of config.py',
                        type=int)
    parser.add_argument('-is',
                        '--image_size',
                        #default=None,
                        help='override the INIT_TRAINER[\'image_size\'] of config.py',
                        type=int)
    parser.add_argument('-bs',
                        '--batch_size',
                        #default=None,
                        help='override the INIT_TRAINER[\'batch_size\'] of config.py',
                        type=int)
    args = parser.parse_args()
    
    if args.net_name is not None:
        INIT_TRAINER['net_name'] = args.net_name
    if args.lr is not None:
        INIT_TRAINER['lr'] = args.lr
    if args.n_epoch is not None:
        INIT_TRAINER['n_epoch'] = args.n_epoch
    if args.num_classes is not None:
        INIT_TRAINER['num_classes'] = args.num_classes
    if args.image_size is not None:
        INIT_TRAINER['image_size'] = args.image_size
    if args.batch_size is not None:
        INIT_TRAINER['batch_size'] = args.batch_size
    ###############################################
    # Set data path & classifier
    if args.mode != 'train-cross' and args.mode != 'inf-cross':
        classifier = VolumeClassifier(**INIT_TRAINER)
        print(get_parameter_number(classifier.net))
    ###############################################
    # 训练模式（train）：加载数据集，进行训练并输出运行时间
    # Training
    ###############################################
    if 'train' in args.mode:
#         csv_path = './csv_file/cub_200_2011.csv_train.csv'
        csv_path = './csv_file/Stanford_Dogs_train.csv'
        label_dict = csv_reader_single(csv_path, key_col='id', value_col='label')
        path_list = list(label_dict.keys())
        # random打乱数据集
        random.seed(1)
        random.shuffle(path_list)
        ###############################################
        if args.mode == 'train-cross':
            for fold in range(1,FOLD_NUM+1):
                print('===================fold %d==================='%(fold))
                if INIT_TRAINER['pre_trained']:
                    INIT_TRAINER['weight_path'] = WEIGHT_PATH_LIST[fold-1]
                classifier = VolumeClassifier(**INIT_TRAINER)
                train_path, val_path = get_cross_validation(path_list, FOLD_NUM, fold)
                SETUP_TRAINER['train_path'] = train_path
                SETUP_TRAINER['val_path'] = val_path
                SETUP_TRAINER['label_dict'] = label_dict
                SETUP_TRAINER['cur_fold'] = fold

                start_time = time.time()
                classifier.trainer(**SETUP_TRAINER)

                print('run time:%.4f' % (time.time() - start_time))
        ###############################################
        elif args.mode == 'train':
            # 划分训练集和验证集
            train_path, val_path = get_cross_validation(path_list, FOLD_NUM, CURRENT_FOLD)
            SETUP_TRAINER['train_path'] = train_path
            SETUP_TRAINER['val_path'] = val_path
            SETUP_TRAINER['label_dict'] = label_dict
            SETUP_TRAINER['cur_fold'] = CURRENT_FOLD

            start_time = time.time()
            classifier.trainer(**SETUP_TRAINER)

            print('run time:%.4f' % (time.time() - start_time))
    
    ###############################################
    # 推理模式（inf）：加载测试集进行推理，生成结果并保存相关信息，如分类报告、特征文件等。
    # Inference
    ###############################################
    elif 'inf' in args.mode:
        # 读取测试集
#         test_csv_path = './csv_file/cub_200_2011.csv_test.csv'
        test_csv_path = './csv_file/Stanford_Dogs_test.csv'
        label_dict = csv_reader_single(test_csv_path, key_col='id', value_col='label')  #
        test_path = list(label_dict.keys()) #[0:32]
         # random打乱数据集
        random.seed(99)
        random.shuffle(test_path)
        print('test len:',len(test_path))
        ###############################################
        # 结果和中间结果存储文件路径
        save_dir = './analysis/result/{}'.format(VERSION)
        feature_dir = './analysis/mid_feature/{}'.format(VERSION)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ###############################################
        if args.mode == 'inf':
            save_path = os.path.join(save_dir,f'fold{str(CURRENT_FOLD)}.csv')
            start_time = time.time()
            if args.save == 'no' or args.save == 'n':
                result, _, _ = classifier.inference(test_path, label_dict)  # 进行推理
                print('run time:%.4f' % (time.time() - start_time))
            else:
                result, feature_in, feature_out = classifier.inference(
                    test_path, label_dict, hook_fn_forward=True)    # 进行推理
                print('run time:%.4f' % (time.time() - start_time))
                ###############################################
                # 存储中间特征
                # save the avgpool output
                '''
                print(feature_in.shape, feature_out.shape)
                feature_save_path = os.path.join(feature_dir,f'fold{str(CURRENT_FOLD)}') # analysis/mid_feature/fold1/name[0]
                if not os.path.exists(feature_save_path):
                    os.makedirs(feature_save_path)
                else:
                    shutil.rmtree(feature_save_path)
                    os.makedirs(feature_save_path)
                for i in range(len(test_path)):
                    name = os.path.basename(test_path[i])
                    feature_path = os.path.join(feature_save_path, name.split(".")[0])
                    save_as_hdf5(feature_in[i], feature_path, 'feature_in')     # 存储中间特征 
                    save_as_hdf5(feature_out[i], feature_path, 'feature_out')   # analysis/mid_feature/fold1/name[0]
                '''
                ###############################################
            # 存储 推理结果
            result['path'] = test_path
            csv_file = pd.DataFrame(result)
            csv_file.to_csv(save_path, index=False)
            ###############################################
            # 存储 report
            cls_report = classification_report(result['true'],result['pred'],output_dict=True)       # 输出模型评估报告
           
            ###############################################
            #fc weight
            if INIT_TRAINER['net_name'].startswith('res') or INIT_TRAINER['net_name'].startswith('wide_res'):
                fc_weight_save_path = os.path.join(save_dir,f'fold{str(CURRENT_FOLD)}_fc_weight.npy')   # analysis/result/fold1_fcweight.npy
                np.save(fc_weight_save_path, classifier.net.state_dict()['fc.weight'].cpu().numpy())    # 2 np.save ? classifier.net. ? . fc.weight
            report_save_path = os.path.join(save_dir,f'fold{str(CURRENT_FOLD)}_report.csv')     
            report_csv_file = pd.DataFrame(cls_report)
            #save as csv  # analysis/result/fold1_report.csv
            report_csv_file.to_csv(report_save_path)

        ###############################################
        elif args.mode == 'inf-cross':
            for fold in range(1,FOLD_NUM+1):
                print('===================fold %d==================='%(fold))
                print('weight path %s'%WEIGHT_PATH_LIST[fold-1])
                INIT_TRAINER['weight_path'] = WEIGHT_PATH_LIST[fold-1]
                classifier = VolumeClassifier(**INIT_TRAINER)
                
                save_path = os.path.join(save_dir,f'fold{str(fold)}.csv')
                start_time = time.time()
                if args.save == 'no' or args.save == 'n':
                    result, _, _ = classifier.inference(test_path, label_dict)
                    print('run time:%.4f' % (time.time() - start_time))
                else:
                    result, feature_in, feature_out = classifier.inference(
                        test_path, label_dict, hook_fn_forward=True)
                    print('run time:%.4f' % (time.time() - start_time))
                    # save the avgpool output
                    print(feature_in.shape, feature_out.shape)
                    feature_save_path = os.path.join(feature_dir,f'fold{str(fold)}')
                    if not os.path.exists(feature_save_path):
                        os.makedirs(feature_save_path)
                    else:
                        shutil.rmtree(feature_save_path)
                        os.makedirs(feature_save_path)
                    for i in range(len(test_path)):
                        name = os.path.basename(test_path[i])
                        feature_path = os.path.join(feature_save_path, name.split(".")[0])
                        save_as_hdf5(feature_in[i], feature_path, 'feature_in')
                        save_as_hdf5(feature_out[i], feature_path, 'feature_out')
                result['path'] = test_path
                csv_file = pd.DataFrame(result)
                csv_file.to_csv(save_path, index=False)
                #report
                cls_report = classification_report(result['true'],result['pred'],output_dict=True)
                #save as csv
                report_save_path = os.path.join(save_dir,f'fold{str(fold)}_report.csv')
                report_csv_file = pd.DataFrame(cls_report)
                report_csv_file.to_csv(report_save_path)
                #fc weight
                if INIT_TRAINER['net_name'].startswith('res') or INIT_TRAINER['net_name'].startswith('wide_res'):
                    fc_weight_save_path = os.path.join(save_dir,f'fold{str(fold)}_fc_weight.npy')
                    np.save(fc_weight_save_path, classifier.net.state_dict()['fc.weight'].cpu().numpy())
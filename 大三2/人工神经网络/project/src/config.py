from utils import get_weight_path,get_weight_list

__all__ = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152","resnext50_32x4d","resnext101_32x8d","resnext101_64x4d","wide_resnet50_2","wide_resnet101_2",
           "vit_b_16","vit_b_32","vit_l_16","vit_l_32","vit_h_14"]

NET_NAME = 'vit_b_16'
VERSION = 'v1.0'
DEVICE = '2'
# Must be True when pre-training and inference
PRE_TRAINED = False
MODEL_PRETRAINED = True
# choice for using _get_net:'1' _get_net_model:'2' _get_net_hub:'3'
MODEL_MOTHOD= '1'
# 1,2,3,4,5
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 5

CUB_TRAIN_MEAN = [0.48560741861744905, 0.49941626449353244, 0.43237713785804116]
CUB_TRAIN_STD = [0.2321024260764962, 0.22770540015765814, 0.2665100547329813]

CKPT_PATH = './ckpt/{}/fold{}'.format(VERSION,CURRENT_FOLD)
WEIGHT_PATH = get_weight_path(CKPT_PATH)
# print(WEIGHT_PATH)

if PRE_TRAINED:
    WEIGHT_PATH_LIST = get_weight_list('./ckpt/{}/'.format(VERSION))
else:
    WEIGHT_PATH_LIST = None

# Arguments when trainer initial
INIT_TRAINER = {
    'net_name':NET_NAME,
    'lr':5e-5, 
    'n_epoch':200,
    'num_classes':200,
    'model_pretrained':MODEL_PRETRAINED,
    'image_size':256,
    'batch_size':32,
    'train_mean':CUB_TRAIN_MEAN,
    'train_std':CUB_TRAIN_STD,
    'num_workers':2,
    'device':DEVICE,
    'pre_trained':PRE_TRAINED,
    'weight_path':WEIGHT_PATH,
    'weight_decay':1e-3,
    'momentum': 0.9,
    'gamma': 0.1,
    'milestones': [30,60,90],
    'T_max':5,
    'use_fp16':True,
    'dropout':0.01,
    'attention_dropout':0.000005,
    'model_method':MODEL_MOTHOD,
    'l1_reg':None,
    'lp_reg':None,
    'mixup_enable':True
 }

# Arguments when perform the trainer 
SETUP_TRAINER = {
    'output_dir':'./ckpt/{}'.format(VERSION),
    'log_dir':'./log/{}'.format(VERSION),
    'optimizer':'AdamW',# Adam,SGD,AdamW
    'loss_fun':'Cross_Entropy',
    'class_weight':None,
    'lr_scheduler':'MultiStepLR'  # ReduceLROnPlateau, MultiStepLR,CosineAnnealingLR,CosineAnnealingWarmRestarts
}


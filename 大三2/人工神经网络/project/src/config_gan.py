from utils import get_weight_path

VERSION = 'v1.0'
DEVICE = 2
# Must be True when pre-training and generating jpg
PRE_TRAINED = True


GEN_PIC = True
if GEN_PIC:
    PRE_TRAINED = True

GEN_PATH = './ckpt/{}_gan/generator.pth'.format(VERSION)
DIS_PATH = './ckpt/{}_gan/discriminator.pth'.format(VERSION)

DATA_PATH = 'gan_dataset/{}/'.format(VERSION)
DATA_NUM = 10

INIT_TRAINER = {
    'image_size':64,
    'encoding_dims':100,
    'batch_size':128,
    'epochs':40,
    'num_workers':1,
    'gen_path':GEN_PATH,
    'dis_path':DIS_PATH
}
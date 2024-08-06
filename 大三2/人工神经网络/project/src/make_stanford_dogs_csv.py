import os
import scipy.io
import pandas as pd

def make_csv_stanford_dogs(input_path, csv_path, mat_file):
    '''
    Make Stanford Dogs csv file.
    根据Stanford Dogs数据集提供的.mat文件生成csv文件
    '''
    
    # 加载 .mat 文件
    data = scipy.io.loadmat(mat_file)
    file_list = data['file_list']

    # 提取路径数据到列表
    paths = [item[0][0] for item in file_list]

    # 创建标签字典
    label_dict = {}
    breed_to_label = {}

    for path in paths:
        breed = path.split('/')[0]
        if breed not in breed_to_label:
            breed_to_label[breed] = len(breed_to_label)
        label_dict[input_path + path] = breed_to_label[breed]

    # 创建 DataFrame
    col = ['id', 'label']
    info = [[key, value] for key, value in label_dict.items()]
    info_data = pd.DataFrame(columns=col, data=info)
    
    # 将 DataFrame 存储为 CSV 文件
    info_data.to_csv(csv_path, index=False)
    print(f"CSV file has been created successfully at: {csv_path}")

if __name__ == "__main__":
    # 创建 csv 文件
    if not os.path.exists('./csv_file'):
        os.makedirs('./csv_file')

    train = False

    input_path = './datasets/Stanford_Dogs/Images/'
    csv_path = './csv_file/Stanford_Dogs_train.csv' if train else './csv_file/Stanford_Dogs_test.csv'
    mat_file = './datasets/Stanford_Dogs/train_list.mat' if train else './datasets/Stanford_Dogs/test_list.mat'

    make_csv_stanford_dogs(input_path, csv_path, mat_file)

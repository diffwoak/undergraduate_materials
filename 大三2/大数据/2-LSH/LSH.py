import random
import numpy as np
import pandas as pd

# 生成Shingle集合
def shingle(text, k):
    shingle_set = []
    for i in range(len(text) - k+1):
        shingle_set.append(text[i:i+k])
    return set(shingle_set)

# 构建词汇表
def build_vocabulary(shingles_list):
    vocabulary = set()
    for shingles in shingles_list:
        vocabulary.update(shingles)
    return list(vocabulary)

# 将Shingles转换为One-hot向量
def shingles_to_onehot(shingles, vocabulary):
    one_hots = []
    for shingle in shingles:
        one_hot = np.zeros(len(vocabulary), dtype=int)
        for s in shingle:
            idx = vocabulary.index(s)
            one_hot[idx] = 1
        one_hots.append(one_hot)
    return one_hots

# 将one-hot向量转化为signatures
def minhash(one_hot_vector, num_hashes):
    signatures = [[],[]]
    # signatures.append([])
    # signatures.append([])
    for _ in range(num_hashes):
        hash_ex = list(range(1, len(one_hot_vector[0])+1))
        random.shuffle(hash_ex)
        for j in range(len(one_hot_vector)):
            for i in range(1, len(one_hot_vector[j])+1):
                idx = hash_ex.index(i)
                if one_hot_vector[j][idx] == 1:
                    signatures[j].append(idx)
                    break
    return signatures

# Locality-Sensitive Hashing
def locality_sensitive_hashing(signatures, bands, rows):
    buckets = []
    for i in range(0, len(signatures), rows):
        band = tuple(signatures[i:i + rows])
        hash_band = hash(band) % (10**9 + 7)
        # if (i // rows) in buckets:
        buckets.append(hash_band)
        # else:
        #     buckets[i // rows] = [hash_band]
    return buckets

def calculate_false_positives_false_negatives(similar_pairs, lsh_pairs, similarity_threshold):
    false_positives = 0
    false_negatives = 0

    for pair in lsh_pairs:
        if pair not in similar_pairs:
            false_positives += 1

    for pair in similar_pairs:
        if pair not in lsh_pairs and compute_similarity(pair) >= similarity_threshold:
            false_negatives += 1
    return false_positives, false_negatives

def compute_similarity(shingle_1,shingle_2):
    intersection = len(shingle_1.intersection(shingle_2))
    union = len(shingle_1.union(shingle_2))
    similarity = intersection / union
    return similarity

def process_data(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    # 查看数据的前几行
    # print(df.head())
    # 访问特定列数据
    ids = df['id']
    questions1 = df['question1']
    questions2 = df['question2']
    is_duplicate = df['is_duplicate']

    false_negatives = 0
    false_positives = 0
    for idx in range(piar_num):
        shingle_list = []
        # print("first text:"+questions1[idx])
        # print("second text:"+questions1[idx])
        shingle_list.append(shingle(questions1[idx],k))
        shingle_list.append(shingle(questions2[idx],k))
        if len(shingle_list[0])==0 or len(shingle_list[1])==0:
            continue
        vocab = build_vocabulary(shingle_list)
        x = shingles_to_onehot(shingle_list,vocab)
        # print("one-hot")
        # print(x)
        s = minhash(x,M)
        # print("signatures:")
        # print(s)
        b1 = locality_sensitive_hashing(s[0], bands, row)
        b2 = locality_sensitive_hashing(s[1], bands, row)
        # print("bucket:")
        # print(b1)
        # print(b2)
        is_candidate = False
        for i in range(bands):
            if b1[i] == b2[i]:
                is_candidate = True
        sim = compute_similarity(shingle_list[0],shingle_list[1])
        if sim >= similarity_threshold and  not is_candidate:
            false_negatives += 1
        if sim < similarity_threshold and is_candidate:
            false_positives += 1
    print(false_negatives)
    print(false_positives)

def main():
    file_path = 'questions.csv'
    process_data(file_path)

if __name__ == "__main__":
    piar_num = 10000
    bands = 20
    row = 15
    M = 300
    k = 10
    similarity_threshold = 0.8
    main()

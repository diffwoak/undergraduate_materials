from itertools import combinations
import json
import time
import pandas as pd
import argparse


# 预处理数据集
# item_mapping: 商品-二进制位数映射
# compressed: 用户-商品的二进制表示
def init_transactions(database):
    compressed = []
    item_mapping = {}
    index = 0
    for transaction in database:
        compressed_transaction = 0
        for item in transaction:
            if item not in item_mapping:
                item_mapping[item] = index
                index += 1
            compressed_transaction |= (1 << item_mapping[item])
        compressed.append(compressed_transaction)
    return compressed, item_mapping

# 初始化生成频繁1项集
def generate_frequent_itemsets(database, min_support, item_mapping):
    item_counts = {}
    for transaction in database:
        for item in transaction:
            if item in item_counts:
                item_counts[item] += 1
            else:
                item_counts[item] = 1
    frequent_itemsets = {1 << item_mapping[item] for item, count in item_counts.items() if count >= min_support}
    return frequent_itemsets

# 判断k候选项集是否满足子集存在于k-1频繁项集中
def has_frequent_subsets(candidate, frequent_itemsets):
    # 获取候选项集中所有置位的位
    items = [i for i in range(candidate.bit_length()) if (candidate & (1 << i))]
    # 生成所有长度为 k-1 的子集
    subsets = combinations(items, len(items) - 1)
    for subset in subsets:
        subset_mask = sum(1 << i for i in subset)
        if subset_mask not in frequent_itemsets:
            return False
    return True

# 生成候选项集
def generate_candidate(frequent_itemsets, k):
    candidates = set()
    frequent_items = list(frequent_itemsets)
    for i in range(len(frequent_items)):
        for j in range(i + 1, len(frequent_items)):
            # 合并两个频繁项集
            itemset1, itemset2 = frequent_items[i], frequent_items[j]
            # 保证前k-2个项相同
            if bin(itemset1 & ((1 << (itemset1.bit_length() - 1)) - 1)) == bin(itemset2 & ((1 << (itemset2.bit_length() - 1)) - 1)):
                candidate = itemset1 | itemset2
                if bin(candidate).count('1') == k and has_frequent_subsets(candidate, frequent_itemsets):
                    candidates.add(candidate)
    return candidates

# 计算组合数
def combines(c, k) :
    x = 1
    y = 1
    for i in range(k) :
        x *= (c - i)
        y *= (i + 1)
    return x / y

# 筛选新的频繁项集
def count_prune(transactions, candidates, k, min_support):
    candidate_counts = {itemset: 0 for itemset in candidates}
    c_len = len(candidates)
    for transaction in transactions:
        tr = [i for i in range(transaction.bit_length()) if (transaction & (1 << i))]
        # if combines(len(tr), k) > c_len:
        if 1:
            for candidate in candidates:
                # 使用位运算检查是否是候选项集的子集
                if (transaction & candidate) == candidate:
                    candidate_counts[candidate] += 1
        # 如果candidates数量更大,则构造transaction的k子集
        else:
            subsets = combinations(tr, k)
            for subset in subsets:
                subset_mask = sum(1 << i for i in subset)
                if subset_mask in candidates:
                    candidate_counts[subset_mask] += 1
    frequent_itemsets = {itemset for itemset, count in candidate_counts.items() if count >= min_support}
    return frequent_itemsets

# 二进制数映射为具体项
def decode_itemset(itemset, item_mapping):
    items = []
    for item, index in item_mapping.items():
        if itemset & (1 << index):
            items.append(item)
    return items

# 保存中间结果
def save_result(frequent_itemsets, item_mapping, min_support, k):
    with open(f'fre{min_support}_{k}.txt', 'w') as file:
            for itemset in frequent_itemsets:
                items = decode_itemset(itemset, item_mapping)
                line = ""
                for i in items:
                    line += i+","
                line += '\n'
                file.write(line)


def apriori(transactions, min_support):
    times = [time.time()]
    # 预处理数据集
    transactions, item_mapping = init_transactions(transactions)
    # 生成频繁1项集
    frequent_itemsets = generate_frequent_itemsets(database, min_support,item_mapping)
    k = 1
    while frequent_itemsets:
        times.append(float(time.time()))
        print(f"k: {k}\tfrequent_itemsets nums: {len(frequent_itemsets)}")
        # save_result(frequent_itemsets,item_mapping, min_support, k)
        k += 1
        # 生成候选项集
        candidates = generate_candidate(frequent_itemsets, k)
        # 计算支持度 筛选出频繁项集
        frequent_itemsets = count_prune(transactions, candidates, k, min_support)
    
    base = times[0]
    for i in range(len(times)):
        times[i] = times[i] - base
    timepd = pd.DataFrame(data = times,columns = ['time'])
    timepd.to_csv(f'times_record_v1_{min_support}.csv') 
    # timepd.to_csv(f'times_record_v2_{min_support}.csv')
    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--t","-t", default=5,type=int, help="The support threshold")
    # args = parser.parse_args()

    # 示例数据
    # database = [
    #     {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T'},
    #     {'A', 'C', 'E', 'G', 'I', 'K', 'M', 'O', 'Q', 'S'},
    #     {'B', 'D', 'F', 'H', 'J', 'L', 'N', 'P', 'R', 'T'},
    #     {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}
    # ]
    # 开始时间
    start_time = time.time()
    min_support = 3
    # min_support = args.t
    # 读取Groceries.json文件
    with open('Groceries.json', 'r') as file:
        data = json.load(file)
        database = [items.split(',') for items in data.values()]

    apriori(database, min_support)

    # 结束时间
    end_time = time.time()
    time = end_time - start_time
    print(f"threshold {min_support} time taken: {time:.4f} s")
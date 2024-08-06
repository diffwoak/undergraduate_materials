from collections import defaultdict
from itertools import combinations
import json
import time
import pandas as pd
from utils import *

# 生成候选项集
def generate_candidate(frequent_itemsets, k):
    candidates = set()
    frequent_items = sorted([bin(t)[:1:-1] for t in frequent_itemsets])
    for i in range(len(frequent_items)):
        item1 = frequent_items[i]
        end = item1[:-1].rfind('1') + 1
        for j in range(i + 1, len(frequent_items)):
            if item1[:end] == frequent_items[j][:end]:
                # 合并两个频繁项集
                candidate = int(item1[::-1],2) | int(frequent_items[j][::-1],2)
                candidates.add(candidate)
            else:
                i = j
                break
    return candidates

# 筛选新的频繁项集
def count_prune(transactions, candidates, k, min_support):
    candidate_counts = {itemset: 0 for itemset in candidates}
    c_len = len(candidates)
    for transaction in transactions:
        tr = [i for i in range(transaction.bit_length()) if (transaction & (1 << i))]
        if combines(len(tr), k) > c_len:
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
    # timepd.to_csv(f'times_record_v1_{min_support}.csv') 
    timepd.to_csv(f'result\\times_record_v3_{min_support}.csv')
    
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
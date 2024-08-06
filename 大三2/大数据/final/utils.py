
from collections import defaultdict


# 计算组合数
def combines(c, k) :
    x = 1
    y = 1
    for i in range(k) :
        x *= (c - i)
        y *= (i + 1)
    return x / y


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
    item_counts = defaultdict(int)
    for transaction in database:
        for item in transaction:
            item_counts[item] += 1
    frequent_itemsets = {1 << item_mapping[item] for item, count in item_counts.items() if count >= min_support}
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
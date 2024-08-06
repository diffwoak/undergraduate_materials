## LSH算法

>  Locality Sensitive Hashing 局部敏感哈希，也叫大规模数据邻近算法

### 实验要求

- 实现LSH算法（从Shingling到Locality-Sensitive Hashing完整步骤）
- 文档数据集对LSH的效果进行测试，设置k=10，相似性阈值0.8，通过调整M、b、r分析LSH的假阴性和假阳性

1. 一般都是直接使用``from datasketch import MinHash, MinHashLSH``调用类完成
2. 要求实现算法说明要编写细节代码

### 思路

- 导入文档 ok
- 实现三大块 [一文读懂局部敏感哈希（LSH）算法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/645577495)
- 主函数拼接 

### 问题

- one-hot没有给到同一个hash

### 参考连接

- [经典算法系列-搞懂大规模数据近邻算法LSH算法原理 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/581008101)
- [【算法】局部敏感哈希 LSH 的 Python 实现_python minhashlsh-CSDN博客](https://blog.csdn.net/qq_36643449/article/details/124882484)

- [局部敏感哈希（LSH）_W24-的博客-CSDN博客](https://blog.csdn.net/qq_39583450/category_10851544.html)



#### code from gpt

```python
import random

# 假设文档集合已经处理成Shingle集合和MinHash签名
documents = [
    {"id": 1, "shingles": set(["shingle1", "shingle2", ...]), "minhash": [hash_val1, hash_val2, ...]},
    {"id": 2, "shingles": set(["shingle3", "shingle4", ...]), "minhash": [hash_val3, hash_val4, ...]},
    ...
]

# LSH算法
def locality_sensitive_hashing(documents, M, b, r, similarity_threshold):
    # 实现LSH算法逻辑
    pass

# 测试LSH效果
M = 100  # MinHash签名数量
b = 10  # 带状区域数量
r = 10  # 每个带状区域包含的MinHash签名数量
similarity_threshold = 0.8  # 相似性阈值

# 调用LSH算法
similar_pairs, false_positives, false_negatives = locality_sensitive_hashing(documents, M, b, r, similarity_threshold)

# 输出测试结果
print("真正相似的文档对：", similar_pairs)
print("LSH判断为相似但实际不相似的文档对：", false_positives)
print("LSH未能判断为相似但实际相似的文档对：", false_negatives)

# 分析假阴性和假阳性情况，并调整参数进行测试

```

```python
import random

# 生成Shingle集合
def generate_shingles(document, k):
    shingles = set()
    words = document.split()
    for i in range(len(words) - k + 1):
        shingle = ' '.join(words[i:i + k])
        shingles.add(shingle)
    return shingles

# MinHash签名
def minhash_signature(shingles, num_hashes):
    signatures = []
    for _ in range(num_hashes):
        hash_func = lambda x: hash(str(x) + str(random.randint(0, 1000)))
        minhash = float('inf')
        for shingle in shingles:
            hash_val = hash_func(shingle)
            minhash = min(minhash, hash_val)
        signatures.append(minhash)
    return signatures

# Locality-Sensitive Hashing
def locality_sensitive_hashing(signatures, bands, rows_per_band):
    buckets = {}
    for i in range(0, len(signatures), rows_per_band):
        band = tuple(signatures[i:i + rows_per_band])
        if band in buckets:
            buckets[band].append(i // rows_per_band)
        else:
            buckets[band] = [i // rows_per_band]
    return buckets

# 示例用法
document1 = "This is a sample document for shingling and LSH implementation"
document2 = "Sample document for testing LSH algorithm"
shingles1 = generate_shingles(document1, 3)
shingles2 = generate_shingles(document2, 3)
signatures1 = minhash_signature(shingles1, 50)
signatures2 = minhash_signature(shingles2, 50)
buckets = locality_sensitive_hashing(signatures1, bands=10, rows_per_band=5)
print(buckets)
```


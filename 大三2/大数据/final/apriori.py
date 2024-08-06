import json
import pandas as pd
import numpy as np
import time

def count_prune(candidates,transactions,threshold):
	if candidates == []:return []
	nums = [0] * len(candidates)
	for transaction in transactions:
		for i,candidate in enumerate(candidates):
			is_support = True
			for item in candidate:
				if item not in transaction:
					is_support = False
					break
			if is_support:
				nums[i]+=1
	positions = [i for i, num in enumerate(nums) if num >= threshold]
	can = [candidates[i] for i in positions]
	candidates = []
	for i in can:
		candidates.append(','.join(i))
	return candidates

def generate_next_candidates(candidates):
	candidates = list(map(lambda i: sorted(i.split(',')), candidates))
	k = len(candidates[0])
	C = []
	for i in range(len(candidates)):
		for j in range(i+1, len(candidates)):
			if candidates[i][:k - 1] == candidates[j][:k - 1] and candidates[i][k - 1] != candidates[j][k - 1]:
				tmp = candidates[i][:k - 1] + sorted([candidates[j][k - 1], candidates[i][k - 1]])
				# 判断所有k项的子集是否在频繁项集中
				subsets = []
				for q in range(len(tmp)):
					t = [tmp[q]]
					tt = sorted(list(set(tmp) - set(t)))
					subsets.append(tt)
				is_fre = True
				for w in subsets:
					if w not in candidates:
						is_fre = False
						break
				if is_fre:
					C.append(tmp)
	return C

def write_candidates(candidates,t,k):
	with open(f'candidates{t}_{k}.txt', 'w') as file:
		for item_set in candidates:
			# 将集合转换为逗号分隔的字符串，并写入文件
			line = item_set + '\n'
			file.write(line)

if __name__ == "__main__":

	start_time = time.time()

	threshold = 3
	# 读取Groceries.json文件
	with open('Groceries.json', 'r') as file:
		data = json.load(file)
	# 创建0-1矩阵
	all_items = set()
	for items_str in data.values():
		items = items_str.split(',')
		all_items.update(items)
	all_items = sorted(list(all_items))
	transactions = []
	for items_str in data.values():
		items = items_str.split(',')
		transaction_vector = [1 if item in items else 0 for item in all_items]
		transactions.append(transaction_vector)
	df = pd.DataFrame(transactions, columns=all_items)
	transactions = [items.split(',') for items in data.values()]
	
	times = [time.time()]
	# 初始化首轮候选项集
	support_series = df.sum(axis=0)
	candidates = list(support_series[support_series > threshold].index)
	
	k = 1
	while candidates != []:
		times.append(float(time.time()))
		print(f'set {k} --nums of candidates: {len(candidates)}')
		k = k + 1
		# write_candidates(candidates,threshold,k)
        # 连接产生新的候选项集
		candidates =generate_next_candidates(candidates)
        # 根据阈值过滤产生频繁项集
		candidates = count_prune(candidates,transactions,threshold)

	base = times[0]
	for i in range(len(times)):
		times[i] = times[i] - base
	timepd = pd.DataFrame(data = times,columns = ['time'])
	timepd.to_csv(f'times_record_base_{threshold}.csv')

	end_time = time.time()
	time = end_time - start_time
	print(f"threshold {threshold} time taken: {time:.4f} s")
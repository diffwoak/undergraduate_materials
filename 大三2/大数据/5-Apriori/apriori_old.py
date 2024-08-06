import json


def count_prune(candidates,transactions,threshold):
	if candidates == []:return []
	nums = [0] * len(candidates)
	for transaction in transactions:
		for i,candidate in enumerate(candidates):
			is_support = True
			for item in candidate:
				if item not in transaction:
					is_support = False
			if is_support:
				nums[i]+=1
	positions = [i for i, num in enumerate(nums) if num > threshold]
	return [candidates[i] for i in positions]
def generate_next_candidates(candidates,L1):
	k = len(candidates[0])
	nums = []
	new_candidates = []
	for candidate in candidates:
		for l in L1:
			if not l.issubset(candidate):# 判断是否可连接
				tmp = candidate.union(l)
				if tmp not in new_candidates:# 不存在相同项，则添加nums记录
					nums.append(1)
					new_candidates.append(tmp)
				else:						# 存在相同项，则更新nums记录
					location = new_candidates.index(tmp)
					nums[location] += 1
	positions = [i for i, num in enumerate(nums) if num == k+1]
	return [new_candidates[i] for i in positions]
def write_candidates(candidates):
	k = len(candidates[0])
	with open(f'candidates_{k}.txt', 'w') as file:
		for item_set in candidates:
			# 将集合转换为逗号分隔的字符串，并写入文件
			line = ','.join(item_set) + '\n'
			file.write(line)

if __name__ == "__main__":
	threshold = 3
	# 读取Groceries.json文件
	with open('Groceries.json', 'r') as file:
		data = json.load(file)
	transactions = [items.split(',') for items in data.values()]
	# 初始化C_1,L_1
	C1 = []
	for transaction in transactions:
		for item in transaction:
			if {item} not in C1:
				C1.append({item})
	print(f"initial candidates:{len(C1)}")
	L1 = count_prune(C1,transactions,threshold)
	print(f"count nums and prune candidates:{len(L1)}")
	candidates = L1
	while candidates != []:
		print(f'L1:{len(L1)}')
		# 生成候选集
		new_C = generate_next_candidates(candidates,L1)
		print(f"generate k+1 candidates:{len(new_C)}")
		if new_C == []:	break
		# 更新频繁集
		new_L = count_prune(new_C,transactions,threshold)
		print(f"count nums and prune candidates:{len(new_L)}")
		if new_L != []:
			candidates = new_L
			# 写入文件
			write_candidates(candidates)
		else:
			break
		# 更新L1，保留只存在最新频繁项集的项，减少连接的搜索空间
		L1 = set()
		for candidate in candidates:
			L1.update(candidate)
		L1 = [{item} for item in L1]
	print(candidates)
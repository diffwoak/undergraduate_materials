from Node import Node,MiddleNode,LastNode

# 迭代器
class Iterator:
    def __init__(self,_collection):
        # 遍历的集合
        self.collection = _collection
        # 当前位置
        self.current_pos = -1

    # 返回下一个遍历元素
    def getNext(self):
        pass
    # 返回是否遍历结束
    def hasMore(self):
        pass
    # 返回当前位置
    def get_pos(self):
        return self.current_pos

# 用于Funny JSON Explorer的迭代器
class FJEIterator(Iterator):
    def __init__(self,_collection):
        super().__init__(_collection)
        # json 文件
        self.json_object = self.collection.load()
        # json转化为Node类对象后的存储列表
        self.nodes = []
        # 记录最大深度
        self.max_level = -1
        self.generate_leafs(self.json_object)

    # json的dict对象转化为Node对象
    def generate_leafs(self, json_object, level = 0):
        if isinstance(json_object, dict):
            # 需要记录当前节点是该level中的开始/中间/结尾节点
            for count, (key, value) in enumerate(json_object.items()):
                # 根据索引判断节点位置
                if count == len(json_object)-1:
                    state = "end"
                elif count == 0:
                    state = "begin"
                else:
                    state = "middle"
                # 创建当前节点
                self.nodes.append(Node(key,level,state))
                index = len(self.nodes) - 1
                ifleaf = self.generate_leafs(value,level+1)
                self.max_level = max(self.max_level,level)
                # 根据ifleaf判断当前节点是否为叶节点,修改当前节点的icon类型
                if ifleaf:
                    # self.nodes[index].set_icon(self.icon_product.getLeafIcon())
                    self.nodes[index] = LastNode(self.nodes[index])
                else:
                    # self.nodes[index].set_icon(self.icon_product.getMiddleIcon())
                    self.nodes[index] = MiddleNode(self.nodes[index])
            return False
        else:
            if json_object is not None:
                self.nodes.append(Node(f'{json_object}'))
            return True

    def getNext(self):
        self.current_pos += 1
        return self.nodes[self.current_pos]

    def hasMore(self):
        return (self.current_pos + 1) < len(self.nodes)

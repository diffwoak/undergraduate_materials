from Iterator import FJEIterator
import json

# 集合
class Collection:
    def __init__(self,file_path):
        self.file_path = file_path

    def load(self):
        pass
    def createIterator(self):
        pass

# 用于Funny JSON Explorer的集合
class FJECollection(Collection):
    def __init__(self,file_path):
        super().__init__(file_path)
        self.json_object = None

    # 加载json为dict类型对象
    def load(self):
        with open(self.file_path, 'r') as file:
            self.json_object = json.load(file)
        return self.json_object
    
    # 创建用于Funny JSON Explorer的迭代器
    def createIterator(self):
        return FJEIterator(self)
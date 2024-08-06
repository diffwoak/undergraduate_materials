# Leaf类
class Node:
    def __init__(self, name, level = -1,state = ""):
        # 节点名称
        self.name = name
        # 节点所在层级
        self.level = level
        # 节点状态 begin:本层级的第一个节点,end:本层级最后一个节点,middle:本层级中间节点
        self.state = state

    def get_state(self):
        return self.state

    def get_level(self):
        return self.level
    
    # 普通节点接受visitor访问
    def accept(self,visitor):
        return visitor.visit_norm(self)

    # 返回该节点表示的字符串
    def draw(self):
        result = ""
        if self.level == -1:
            result += ":"
        result += f"{self.name}"
        return result

class MiddleNode(Node):
    def __init__(self,node):
        self.__dict__.update(node.__dict__)

    # 中间节点接受visitor访问
    def accept(self,visitor):
        return visitor.visit_middle(self)

class LastNode(Node):
    def __init__(self,node):
        self.__dict__.update(node.__dict__)

    # 叶子节点接受visitor访问
    def accept(self,visitor):
        return visitor.visit_last(self)
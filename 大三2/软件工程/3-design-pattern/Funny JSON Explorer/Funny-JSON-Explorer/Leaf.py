# Leaf类
class Leaf:
    def __init__(self, name, level = -1, icon = " ",state = ""):
        # 节点名称
        self.name = name
        # 节点所在层级
        self.level = level
        # 节点前的icon图标
        self.icon = icon
        # 节点状态 begin:本层级的第一个节点,end:本层级最后一个节点,middle:本层级中间节点
        self.state = state

    def set_icon(self,icon):
        self.icon = icon

    def get_state(self):
        return self.state

    def get_level(self):
        return self.level

    # 返回该节点表示的字符串
    def draw(self):
        result = ""
        if self.level == -1:
            result += ":"
        result += f"{self.icon}{self.name}"
        return result
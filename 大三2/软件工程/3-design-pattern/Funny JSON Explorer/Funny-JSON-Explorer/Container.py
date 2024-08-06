from Leaf import Leaf
# Container类
# 根据json文件和icon_family生成节点Leafs对象
class Container:
    def __init__(self,icon_product) -> None:
        self.icon_product = icon_product
        self.leafs = []
        self.max_level = 0

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
                self.leafs.append(Leaf(key,level," ",state))
                index = len(self.leafs) - 1
                ifleaf = self.generate_leafs(value,level+1)
                self.max_level = max(self.max_level,level)
                # 根据ifleaf判断当前节点是否为叶节点,修改当前节点的icon类型
                if ifleaf:
                    self.leafs[index].set_icon(self.icon_product.getLeafIcon())
                else:
                    self.leafs[index].set_icon(self.icon_product.getMiddleIcon())
            return False
        else:
            if json_object is not None:
                self.leafs.append(Leaf(f'{json_object}'))
            return True

    def draw(self):
        pass

# 以下是继承Contianer不同类对象,Container父类统一实现了节点的生成,子类只需根据Container类型实现不同的draw函数

# 树形结构Container类
class TreeContainer(Container): 
    def draw(self):
        in_level = [0] * (self.max_level+1)
        for leaf in self.leafs:
            state = leaf.get_state() 
            level = leaf.get_level()
            if level >= 0:
                if state == "begin":
                    in_level[level] = 1
                
                print()
                for i in range(len(in_level)):
                    if i > level: break
                    if in_level[i] == 0 and i < level:
                        print("   ",end="")
                    else:
                        if i == level:
                            if state == "end":
                                print("└─",end="")
                            else:
                                print("├─",end="")
                        else:
                            print("│  ",end="")     
                if state == "end":
                    in_level[level] = 0
            print(leaf.draw(),end="")

# 矩形结构Container类
class RectangleContainer(Container):
    def draw(self):
        width = 43
        leave = 0
        for count,leaf in enumerate(self.leafs):
            level = leaf.get_level()
            if level >= 0:
                # 补充矩阵剩余部分
                for i in range(leave):
                    if i < leave - 1:
                        print("─",end="")
                    else:
                        if count == 1:
                            print("┐")
                        else:
                            print("┤")
                # 绘制结点前的边
                for i in range(self.max_level+1):
                    if i > level: break
                    # 第一行
                    if i == 0 and count == 0:
                        print("┌─",end="")
                    # 最后一行
                    elif count == len(self.leafs)-1:
                        if i == 0:
                            print("└─",end="")
                        else:
                            print("─┴─",end="")
                    # 中间行
                    else:
                        if i < level:
                            print("│  ",end="")
                        else:
                            print("├─",end="") 
                # 此行剩余边长数量
                leave = width - 3 * (level+1)
            name = leaf.draw()
            print(leaf.draw(),end="")
            leave -= (len(name) )
        # 补充最后一行的剩余边
        for i in range(leave):
            if i < leave - 1:
                print("─",end="")
            else: print("┘")

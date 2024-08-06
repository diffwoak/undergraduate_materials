
# 访问者类
class Visitor:
    def __init__(self,icon):
        # icon-family 类型
        self.icon = icon
        # 中间节点的icon
        self.middle_icon = None
        # 叶子节点的icon
        self.last_icon = None

        # 根据icon-family类型设置icon,添加新的icon-family类型通过新增方法实现
        if self.icon == "poker-face":
            self.set_icon_pokerface()
        elif  self.icon == "myicon":
            self.set_icon_my()
        else:
            raise ValueError("Unsupported icon family")

    # 设置icon-family为pokerface类型
    def set_icon_pokerface(self):
        self.middle_icon = "♢"
        self.last_icon = "♤"
    
    # 设置icon-family为my自定义类型
    def set_icon_my(self):
        self.middle_icon = "▶"
        self.last_icon = "▷" 
    
    # 访问中间节点加上中间节点icon
    def visit_middle(self,node):
        return self.middle_icon + node.draw()
    
    # 访问叶子节点加上叶子节点icon
    def visit_last(self,node):
        return self.last_icon + node.draw()
    
    # 访问普通节点不加icon
    def visit_norm(self,node):
        return node.draw()
    
    # 对整体进行访问
    def visit_all(self,Iterator):
        pass


# Tree style的访问者
class TreeVisitor(Visitor):
    def visit_all(self,iterator):
        in_level = [0] * (iterator.max_level+1)
        # for node in iterator.nodes:
        while (iterator.hasMore()):
            node = iterator.getNext()
            state = node.get_state() 
            level = node.get_level()
            name = node.accept(self)
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
            print(name,end="")

# Rectangle style的访问者
class RectangleVisitor(Visitor):
    def visit_all(self,iterator):
        width = 43
        leave = 0
        # for count,node in enumerate(iterator.nodes):
        while (iterator.hasMore()):
            node = iterator.getNext()
            count = iterator.get_pos()
            level = node.get_level()
            name = node.accept(self)
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
                for i in range(iterator.max_level+1):
                    if i > level: break
                    # 第一行
                    if i == 0 and count == 0:
                        print("┌─",end="")
                    # 最后一行
                    elif count == len(iterator.nodes)-1:
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
            print(name,end="")
            leave -= (len(name))
        # 补充最后一行的剩余边
        for i in range(leave):
            if i < leave - 1:
                print("─",end="")
            else: print("┘")
import argparse

from Visitor import TreeVisitor,RectangleVisitor
from Collection import FJECollection

# Funny JSON Explorer 类
# 初始化赋值Style和icon-Family, 调用创建对应的 IconFactory 和 Container 类对象
class FunnyJsonExplorer:
    def __init__(self,style,icon,file_path):
        self.style = style
        self.icon = icon
        self.file_path = file_path
    
    def work(self):
        # style 创建visitor访问者对象
        if self.style == "tree":
            visitor = TreeVisitor(self.icon)
        elif self.style == "rectangle":
            visitor = RectangleVisitor(self.icon)
        else:
            raise ValueError("Unsupported style")
        
        # 创建collection集合对象
        collection = FJECollection(self.file_path)
        # 创建iterator迭代器对象
        iter = collection.createIterator()
        # visit使用迭代器进行整体访问
        visitor.visit_all(iter)

if __name__ == "__main__":
    # 输入命令行参数
    parser = argparse.ArgumentParser(description="Funny JSON Explorer")
    parser.add_argument('-f', '--file', type=str, required=True, help='Path to the JSON file')
    parser.add_argument('-s', '--style', type=str, required=True, choices=['tree', 'rectangle'], help='Style of the output (tree or rectangle)')
    parser.add_argument('-i', '--icon', type=str, required=True, choices=['poker-face','myicon'], help='Icon family to use (e.g., poker-face)')

    args = parser.parse_args()

    # 创建 FunnyJsonExplorer 对象
    Explorer = FunnyJsonExplorer(args.style,args.icon,"test.json")
    # Explorer = FunnyJsonExplorer("rectangle","myicon","test.json")
    Explorer.work()

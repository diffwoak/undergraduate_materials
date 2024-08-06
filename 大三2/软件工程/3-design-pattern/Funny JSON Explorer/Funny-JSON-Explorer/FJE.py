import argparse
import json
from Icon import PokerFaceIconFactory,MyIconFactory
from Container import TreeContainer,RectangleContainer

# Funny JSON Explorer 类
# 初始化赋值Style和icon-Family, 调用创建对应的 IconFactory 和 Container 类对象
class FunnyJsonExplorer:
    def __init__(self,style,icon):
        self.style = style
        self.icon = icon
        self.json_object = None

    def load(self,file_path):
        with open(file_path, 'r') as file:
            self.json_object = json.load(file)
    
    def show(self):
        # 创建icon_family工厂类
        if self.icon == "poker-face":
            icon_factory = PokerFaceIconFactory()
        elif  self.icon == "myicon":
            icon_factory = MyIconFactory()
        else:
            raise ValueError("Unsupported icon family")

        icon_product = icon_factory.createIconProduct()

        # style 创建container对象
        if self.style == "tree":
            container = TreeContainer(icon_product)
        elif self.style == "rectangle":
            container = RectangleContainer(icon_product)
        else:
            raise ValueError("Unsupported style")
        
        # 根据JSON数据生成节点
        container.generate_leafs(self.json_object)
        # 输出展示
        container.draw()    

if __name__ == "__main__":
    # 输入命令行参数
    parser = argparse.ArgumentParser(description="Funny JSON Explorer")
    parser.add_argument('-f', '--file', type=str, required=True, help='Path to the JSON file')
    parser.add_argument('-s', '--style', type=str, required=True, choices=['tree', 'rectangle'], help='Style of the output (tree or rectangle)')
    parser.add_argument('-i', '--icon', type=str, required=True, choices=['poker-face','myicon'], help='Icon family to use (e.g., poker-face)')

    args = parser.parse_args()

    # 创建 FunnyJsonExplorer 对象
    Explorer = FunnyJsonExplorer(args.style,args.icon)
    # Explorer = FunnyJsonExplorer("rectangle","myicon")
    Explorer.load("test.json")
    Explorer.show()

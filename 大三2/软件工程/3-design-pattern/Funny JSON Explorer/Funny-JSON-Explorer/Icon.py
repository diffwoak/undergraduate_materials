# 图标族 产品类
class IconProduct:
    def getMiddleIcon(self):
        pass
    def getLeafIcon(self):
        pass

class PokerFaceIconProduct(IconProduct):
    def getMiddleIcon(self):
        return "♢"
    def getLeafIcon(self):
        return "♤"

class MyIconProduct(IconProduct):
    def getMiddleIcon(self):
        return "▶"
    def getLeafIcon(self):
        return "▷" 

# 图标族 工厂类
class IconFactory:
    def createIconProduct (self):
        pass

class PokerFaceIconFactory(IconFactory):
    def createIconProduct (self):
        return PokerFaceIconProduct()
    
class MyIconFactory(IconFactory):
    def createIconProduct (self):
        return MyIconProduct()
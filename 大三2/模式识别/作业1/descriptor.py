from harris import calculate_harris,select_points,draw_with_points
from PIL import Image
from numpy import *
from pylab import *
import cv2

import numpy as np

def match_descriptors(des1, des2, threshold=0.8):
    matches = []  # 存储匹配的描述子索引对
    # 计算两组描述子之间的欧氏距离
    for i, d1 in enumerate(des1):
        best_index = -1
        best_distance = float('inf')
        second_distance = float('inf')

        for j, d2 in enumerate(des2):
            distance = np.linalg.norm(d1 - d2)  # 计算欧氏距离

            if distance < best_distance:
                second_distance = best_distance
                best_distance = distance
                best_index = j
            elif distance < second_distance:
                second_distance = distance

        # 检查最佳距离和次佳距离之间的比值是否小于阈值
        if best_distance < threshold * second_distance:
            matches.append((i, best_index))

    return matches


def get_hogs(img_grey,coords):
	# 创建HOG描述符
	winSize = (16, 16)
	blockSize = (8, 8)
	blockStride = (4, 4)
	cellSize = (4, 4)
	nbins = 9
	hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
	# 提取HOG特征
	hogs = []
	for [x,y] in coords:
		roi = img_grey[x - winSize[0] // 2:x + winSize[0] // 2, y - winSize[1] // 2:y + winSize[1] // 2]
		if roi.shape[0] != winSize[0] or roi.shape[1] != winSize[1]:
			continue
		h = hog.compute(roi)
		hogs.append(h)
	return hogs

def get_sifts(img_grey,coords):
	# 创建SIFT对象
	sift = cv2.SIFT_create()
	# 提取SIFT特征
	kp, des = sift.detectAndCompute(img_grey, None)
	# 提取角点对应的SIFT特征
	sifts = []
	for [x,y] in coords:
		# 在SIFT特征中查找距离最近的关键点
		closest_kp = min(kp, key=lambda k: ((k.pt[1] - x) ** 2 + (k.pt[0] - y) ** 2) ** 0.5)
		# 获取该关键点的特征描述符
		index = kp.index(closest_kp)
		sifts.append(des[index])
	return sifts

def show_match(img1,img2,coord1,coord2,des1,des2,out_path=None):
	
	'''
	直接是用SIFT的关键点配对
	sift = cv2.SIFT_create()
	kp1, des1 = sift.detectAndCompute(img1, None)
	kp2, des2 = sift.detectAndCompute(img2, None)
	'''
	matches = match_descriptors(des1, des2)
	kp1 = [cv2.KeyPoint(float(y), float(x), 1) for (x, y) in coord1]
	kp2 = [cv2.KeyPoint(float(y), float(x), 1) for (x, y) in coord2]
	
	match_img = cv2.drawMatches(img1, kp1, img2, kp2, [cv2.DMatch(_[0], _[1], 0) for _ in matches], None)
	match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
	if out_path != None:
		cv2.imwrite(out_path, match_img)
	cv2.imshow('Matches', match_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def get_pic_info(input_path,hog_sift):
	image = cv2.imread(input_path)
	img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)	#rgb
	img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	#grey
	harrisim = calculate_harris(img_grey)
	coords = select_points(harrisim, 10,0.1)
	if hog_sift =='hog':
		des = get_hogs(img_grey,coords)
	else:
		des = get_sifts(img_grey,coords)
	return img,coords,des

if __name__=="__main__":
	# hog or sift
	hog_sift = 'hog'
	img1, coords_1, des_1 = get_pic_info('images/uttower1.jpg',hog_sift)
	img2, coords_2, des_2 = get_pic_info('images/uttower2.jpg',hog_sift)

	show_match(img1,img2,coords_1,coords_2,des_1,des_2,"results/uttower_match.png")


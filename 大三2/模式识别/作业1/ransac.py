from descriptor import match_descriptors,get_hogs,get_sifts,show_match,get_pic_info
from harris import calculate_harris,select_points,draw_with_points
from PIL import Image
from numpy import *
from pylab import *
import cv2
import numpy as np

def Panorama_uttower(hog_sift = None):
	img1, coords_1, des_1 = get_pic_info('images/uttower1.jpg',hog_sift)
	img2, coords_2, des_2 = get_pic_info('images/uttower2.jpg',hog_sift)
	# show_match(img1,img2,coords_1,coords_2,des_1,des_2)
	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]
	# hog or sift
	if hog_sift != None:
		matches = match_descriptors(des_1, des_2)
		kp1 = np.array([(coords_1[m[0]][1],coords_1[m[0]][0]) for m in matches],dtype=np.float32)
		kp2 = np.array([(coords_2[m[1]][1],coords_2[m[1]][0]) for m in matches],dtype=np.float32)
	else:
		sift = cv2.SIFT_create()
		kp_1, des1 = sift.detectAndCompute(img1, None)
		kp_2, des2 = sift.detectAndCompute(img2, None)
		matches = match_descriptors(des1, des2)
		kp1 = np.array([kp_1[m[0]].pt for m in matches],dtype=np.float32)
		kp2 = np.array([kp_2[m[1]].pt for m in matches],dtype=np.float32)

	affine_matrix, _ = cv2.findHomography(kp2, kp1, cv2.RANSAC, ransacReprojThreshold=3)
	panorama = cv2.warpPerspective(img2, affine_matrix, (w1+w2, h1))
	panorama[0:h1, 0:w1] = img1
	panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
	if hog_sift == 'hog':
		cv2.imwrite("results/uttower_stitching_hog.png", panorama)
	elif hog_sift == 'sift':
		cv2.imwrite("results/uttower_stitching_sift.png", panorama)
	else:
		cv2.imwrite("results/uttower_stitching_sift_all.png", panorama)
	# 显示全景图
	cv2.imshow('Panorama', panorama)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def Panorama_yosemite(hog_sift = None):
	img1, coords_1, des_1 = get_pic_info('images/yosemite1.jpg',hog_sift)
	img2, coords_2, des_2 = get_pic_info('images/yosemite2.jpg',hog_sift)
	img3, coords_3, des_3 = get_pic_info('images/yosemite3.jpg',hog_sift)
	img4, coords_4, des_4 = get_pic_info('images/yosemite4.jpg',hog_sift)
	show_match(img1,img2,coords_1,coords_2,des_1,des_2)
	show_match(img2,img3,coords_2,coords_3,des_2,des_3)
	show_match(img3,img4,coords_3,coords_4,des_3,des_4)
	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]
	h3, w3 = img3.shape[:2]
	h4, w4 = img4.shape[:2]

	# hog or sift
	if hog_sift != None:
		matches_12 = match_descriptors(des_1, des_2,0.85)
		matches_23 = match_descriptors(des_2, des_3,0.85)
		matches_34 = match_descriptors(des_3, des_4,0.85)
		# 将第3张图片固定
		kp12_1 = np.array([(coords_1[m[0]][1],coords_1[m[0]][0]) for m in matches_12],dtype=np.float32)
		kp12_2 = np.array([(coords_2[m[1]][1],coords_2[m[1]][0]) for m in matches_12],dtype=np.float32)
		kp23_2 = np.array([(coords_2[m[0]][1],coords_2[m[0]][0]) for m in matches_23],dtype=np.float32)
		kp23_3 = np.array([(coords_3[m[1]][1]+w1+w2,coords_3[m[1]][0]) for m in matches_23],dtype=np.float32)
		kp34_3 = np.array([(coords_3[m[0]][1]+w1+w2,coords_3[m[0]][0]) for m in matches_34],dtype=np.float32)
		kp34_4 = np.array([(coords_4[m[1]][1],coords_4[m[1]][0]) for m in matches_34],dtype=np.float32)

	else:
		sift = cv2.SIFT_create()
		kp_1, des_1 = sift.detectAndCompute(img1, None)
		kp_2, des_2 = sift.detectAndCompute(img2, None)
		kp_3, des_3 = sift.detectAndCompute(img3, None)
		kp_4, des_4 = sift.detectAndCompute(img4, None)
		matches_12 = match_descriptors(des_1, des_2,0.9)
		matches_23 = match_descriptors(des_2, des_3,0.9)
		matches_34 = match_descriptors(des_3, des_4,0.9)
		kp12_1 = np.array([kp_1[m[0]].pt for m in matches_12],dtype=np.float32)
		kp12_2 = np.array([kp_2[m[1]].pt for m in matches_12],dtype=np.float32)
		kp23_2 = np.array([kp_2[m[0]].pt for m in matches_23],dtype=np.float32)
		kp23_3 = np.array([(kp_3[m[1]].pt[0]+w1+w2,kp_3[m[1]].pt[1]) for m in matches_23],dtype=np.float32)
		kp34_3 = np.array([(kp_3[m[0]].pt[0]+w1+w2,kp_3[m[0]].pt[1]) for m in matches_34],dtype=np.float32)
		kp34_4 = np.array([kp_4[m[1]].pt for m in matches_34],dtype=np.float32)
	
	# 计算图二图四的变换矩阵
	affine_matrix_2, _ = cv2.findHomography(kp23_2, kp23_3, cv2.RANSAC, ransacReprojThreshold=3)
	affine_matrix_4, _ = cv2.findHomography(kp34_4, kp34_3, cv2.RANSAC, ransacReprojThreshold=3)
	# 先计算出图二的变换矩阵,得到图二关键点转换后的位置,进而得到图一的变换矩阵
	kp12_2 = np.hstack((kp12_2, np.ones((len(kp12_2), 1))))
	transform_kp12_2 = np.dot(affine_matrix_2, kp12_2.T).T
	kp12_2 = transform_kp12_2[:, :2] / transform_kp12_2[:, 2:]
	affine_matrix_1, _ = cv2.findHomography(kp12_1, kp12_2, cv2.RANSAC, ransacReprojThreshold=3)
	# 拼接
	panorama1 = cv2.warpPerspective(img1, affine_matrix_1, (w1+w2+w3+w4, h1))
	panorama2 = cv2.warpPerspective(img2, affine_matrix_2, (w1+w2+w3+w4, h1))
	panorama4 = cv2.warpPerspective(img4, affine_matrix_4, (w1+w2+w3+w4, h1))
	panorama4[panorama1 > 0] = panorama1[panorama1 > 0]
	panorama4[panorama2 > 0] = panorama2[panorama2 > 0]
	panorama4[0:h1, w1+w2:w1+w2+w3] = img3

	panorama4 = cv2.cvtColor(panorama4, cv2.COLOR_BGR2RGB)
	# cv2.imwrite("results/yosemite_stitching.png", panorama4)

	if hog_sift == 'hog':
		cv2.imwrite("results/yosemite_stitching_hog.png", panorama4)
	elif hog_sift == 'sift':
		cv2.imwrite("results/yosemite_stitching_sift.png", panorama4)
	else:
		cv2.imwrite("results/yosemite_stitching.png", panorama4)


if __name__=="__main__":
	# 拼接2张uttower图片
	Panorama_uttower('hog')
	# 拼接4张yosemite图片
	Panorama_yosemite('hog')


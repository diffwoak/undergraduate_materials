from PIL import Image
from numpy import *
from pylab import *
import cv2

def calculate_harris(im, ksize = 3, k = 0.06):
	# compute Ix Iy
    Ix = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize)
    Iy = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize)
	# compute A B C
    A = cv2.GaussianBlur(Ix * Ix,(ksize,ksize),0.5,0.5)
    C = cv2.GaussianBlur(Ix * Iy,(ksize,ksize),0.5,0.5)
    B = cv2.GaussianBlur(Iy * Iy,(ksize,ksize),0.5,0.5)
	
    det = A * B - C ** 2
    tr = A + B
	
    return det - k*tr

def select_points(harrisim, min_dist = 10, threshold = 0.1):
	# 阈值处理
	harris_t1 = (harrisim > harrisim.max() * threshold) * 1
	# harris_t1 = (harrisim > 1000) * 1
	coords = array(harris_t1.nonzero()).T
	values_t1 = [harrisim[c[0], c[1]] for c in coords]
	index = argsort(values_t1)

	allowed_locations = zeros(harrisim.shape)
	allowed_locations[min_dist: -min_dist, min_dist: -min_dist] = 1
	
	harris_t2 = []
	for i in index:
		if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
			harris_t2.append(coords[i])
			allowed_locations[(coords[i, 0] - min_dist) : (coords[i, 0] + min_dist), (coords[i, 1] - min_dist) : (coords[i, 1] + min_dist)] = 0
	return harris_t2

def draw_with_points(image, coords,output_path):
	# 绘制点到图像上
	for c in coords:
		cv2.circle(image, (int(c[1]), int(c[0])), 2, (0, 0, 255), -1) 
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# 保存图像
	cv2.imwrite(output_path, image)
	# 显示图像
	cv2.imshow('Image with Points', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	
if __name__=="__main__":
	input_path = 'images/sudoku.png'
	output_path = 'results/sudoku_keypoints.png'
	image = cv2.imread(input_path)

	img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	harrisim = calculate_harris(img_grey)
	coords = select_points(harrisim, 10)

	img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	draw_with_points(img_rgb, coords,output_path)


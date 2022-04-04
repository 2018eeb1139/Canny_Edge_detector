import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim

# defining the canny detector function

def non_maximum_suppress(grad_mag,direction):

	#dimensions of the input image
	row,col = grad_mag.shape
	
	# Looping through every pixel of the grayscale image
	for i in range(col):
		for j in range(row):
			
			grad_ang = direction[j, i]
			grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang)
			
			# selecting the neighbours of the target pixel
			# according to the gradient direction
			# In the x axis direction
			if grad_ang<= 22.5:
				nx_1, ny_1 = i-1, j
				nx_2, ny_2 = i + 1, j
			
			# top right (diagonal-1) direction
			elif grad_ang>22.5 and grad_ang<=(22.5 + 45):
				nx_1, ny_1 = i-1, j-1
				nx_2, ny_2 = i + 1, j + 1
			
			# In y-axis direction
			elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90):
				nx_1, ny_1 = i, j-1
				nx_2, ny_2 = i, j + 1
			
			# top left (diagonal-2) direction
			elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135):
				nx_1, ny_1 = i-1, j + 1
				nx_2, ny_2 = i + 1, j-1
			
			# Restarting the cycle
			elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180):
				nx_1, ny_1 = i-1, j
				nx_2, ny_2 = i + 1, j
			
			# Non-maximum suppression step
			if col>nx_1>= 0 and row>ny_1>= 0:
				if grad_mag[j, i]<grad_mag[ny_1, nx_1]:
					grad_mag[j, i]= 0
					continue

			if col>nx_2>= 0 and row>ny_2>= 0:
				if grad_mag[j, i]<grad_mag[ny_2, nx_2]:
					grad_mag[j, i]= 0

	return grad_mag

def Canny_detector(input_image):
	
	# conversion of image to grayscale
	image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
	
	# Noise reduction step
	gauss_image = cv2.GaussianBlur(image, (5, 5), 1.4)
	
	# Calculation the gradients
	sobelx = cv2.Sobel(np.float32(gauss_image), cv2.CV_64F, 1, 0, 3)
	sobely = cv2.Sobel(np.float32(gauss_image), cv2.CV_64F, 0, 1, 3)
	
	# Conversion of Cartesian coordinates to polar
	grad_mag,direction = cv2.cartToPolar(sobelx, sobely, angleInDegrees = True)
	
	# setting the minimum and maximum thresholds
	# for double thresholding
	grad_mag_max = np.max(grad_mag)
	weak_threshold = grad_mag_max * 0.1
	strong_threshold = grad_mag_max * 0.5
	
	# getting the dimensions of the input image
	row,col = image.shape

	#Non Maximum Suppression
	grad_mag = non_maximum_suppress(grad_mag,direction)

	ids = np.zeros_like(input_image)
	# cv2.imshow('sample_image',mag)
	# double thresholding step
	for i_x in range(col):
		for i_y in range(row):
	
			mag = grad_mag[i_y, i_x]
			if mag<weak_threshold:
				grad_mag[i_y, i_x]= 0
			elif strong_threshold>mag>= weak_threshold:
				ids[i_y, i_x]= 1
			else:
				ids[i_y, i_x]= 2
	
	# cv2.imshow('sample_image',mag)
	# finally returning the magnitude of
	# gradients of edges
	return grad_mag

# Getting the respective images
input_image = cv2.imread('1.png')
# input_image = cv2.imread('2.png')
# input_image = cv2.imread('3.png')

# calling the designed function for finding edges
canny_edge_image = Canny_detector(input_image)
main_canny_image = cv2.Canny(input_image,50,150)
(value, differ) = compare_ssim(canny_edge_image, main_canny_image, full=True)
differ = (differ * 255).astype("uint8")
print("Image_SSIM: {}".format(value))

cv2.imshow('canny_edge_image',canny_edge_image)
cv2.imshow('main_canny_image',main_canny_image)
cv2.waitKey()
cv2.destroyAllWindows()
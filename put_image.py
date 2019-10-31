import numpy as np
import cv2
import math
import imutils
from PIL import Image

def put_image_in_position(img,x1,y1,x2,y2):
	text = "Fuck Coffee"
	font = cv2.FONT_HERSHEY_SIMPLEX
	fontScale = 0.5
	color = (255, 0, 255)
	thickness = 1
	
	textsize = cv2.getTextSize(text,font, fontScale, thickness)[0]
#	print(textsize)
# I'm not sure the reason of dividing into 4 but according to fn.py cor_x, cor_y
	x = int((x1+x2)/4)
	y = int((y1+y2)/4)
# For aligning in the middle of the object
	X = x - int(textsize[0]/2)
	Y = y
#	print(X, Y)

	height,width = img.shape[:2]
#	print(img.shape)
	size = height, width, 3
	black_background = np.zeros(size, dtype=np.uint8)
	cv2.putText(black_background, text, (X, Y), font, fontScale, color, thickness, cv2.LINE_AA)
# Defining degree
	if x2 - x1 == 0:
		tan = 0
	else :
		tan = (y2-y1)/(x2-x1)
	degree = int(math.degrees(math.atan(tan)))
	only_text_ground = black_background

# Rotation clock must be considered
	rotated = imutils.rotate(only_text_ground, -degree)
	result = cv2.bitwise_or(img, rotated)
#	result = cv2.addWeighted(img, alpha, rotated, 1-alpha,0)	

	return result


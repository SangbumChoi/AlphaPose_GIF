import numpy as np
import cv2
import math
import imutils

#n==1,2 is both eye

def put_text_in_face(img,part_line):
	text = "Fuck Coffee"
	font = cv2.FONT_HERSHEY_TRIPLEX
	fontScale = 0.5
	color = (255, 0, 255)
	thickness = 1
	textsize = cv2.getTextSize(text,font, fontScale, thickness)[0]

# I'm not sure the reason of dividing into 4 but according to fn.py cor_x, cor_y
#cor means coordinate point
	X0 = part_line.get(0)[0]
	Y0 = part_line.get(0)[1]
	X1 = part_line.get(1)[0]
	Y1 = part_line.get(1)[1]
	X2 = part_line.get(2)[0]
	Y2 = part_line.get(2)[1]
	X3 = part_line.get(3)[0]
	Y3 = part_line.get(3)[1]
	X4 = part_line.get(4)[0]
	Y4 = part_line.get(4)[1]
	x = int((X0+X1+X2+X3+X4)/5)
	y = int((Y0+Y1+Y2+Y3+Y4)/5)
# For aligning in the middle of the object
	X = x - int(textsize[0]/2)
	Y = y + int(textsize[1]/2)
#	print(X, Y)

	height,width = img.shape[:2]
#	print(img.shape)
	size = height, width, 3
	black_background = np.zeros(size, dtype=np.uint8)
	cv2.putText(black_background, text, (X, Y), font, fontScale, color, thickness, cv2.LINE_AA)
# Defining degree
	if X2 - X1 == 0:
		tan = 90
	else :
		tan = (Y2-Y1)/(X2-X1)
	degree = int(math.degrees(math.atan(tan)))
	only_text_ground = black_background

# Rotation clock must be considered
	rotated = imutils.rotate(only_text_ground, -degree)
	result = cv2.bitwise_or(img, rotated)
#	result = cv2.addWeighted(img, alpha, rotated, 1-alpha,0)	

	return result


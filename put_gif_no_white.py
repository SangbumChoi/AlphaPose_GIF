from gif_properties import convert_gif_to_frames
import cv2
import numpy as np
import sys
np.set_printoptions(threshold=10000)

def put_gif(im_name, img, X1, Y1, scale, file_name):
	#need more thing in im_name
	scale_percent = scale
	gif = cv2.VideoCapture('{}'.format(file_name))
	gif_to_lists = convert_gif_to_frames(gif)
#	print(img.shape)
	(height, width) = img.shape[:2]
#	transparency code
#	img = np.dstack([img, np.ones((height,width), dtype = "uint8")*255])

	im_number = im_name.split('.')[0]
#	print(int(im_number), int(gif_to_lists.size))
#	I'm not sure why we have to decrease one amount of value
	none, gif_number = divmod(int(im_number), int(gif_to_lists.size-1))
#	print(im_number, gif_to_lists.size, gif_number)
#	print(gif_to_lists[83])
	gif_affect = np.array(gif_to_lists[gif_number])

	(B, G, R) = cv2.split(gif_affect)
	A = cv2.bitwise_not(B) | cv2.bitwise_not(G) | cv2.bitwise_not(R)
	B = cv2.bitwise_and(B, B, mask=A)
	G = cv2.bitwise_and(G, G, mask=A)
	R = cv2.bitwise_and(R, R, mask=A)
#	transparency code
#	print(A, A.shape)	
#	gif_affect = cv2.merge([B,G,R,A])
	gif_affect = cv2.merge([B,G,R])

#	print(gif_affect)

	(gif_height, gif_width) = gif_affect.shape[:2]

	gif_height = int(gif_height * scale_percent / 100)
	gif_width = int(gif_width * scale_percent / 100)
	dim = (gif_width, gif_height)

	gif_affect = cv2.resize(gif_affect, dim)

#	print(gif_affect)

	overlay = np.zeros((height, width, 3), dtype="uint8")
	start_Y = int(Y1-gif_height/2)
	start_X = int(X1-gif_width/2)
#	print(Y1, X1, gif_height, gif_width, start_Y, start_X, height, width)
# 	Exceptioin for whether if gif is out of bound
	if start_Y > 0 and start_X > 0 and start_Y+gif_height < height and start_X+gif_width < width:
		overlay[start_Y:start_Y+gif_height, start_X:start_X+gif_width] = gif_affect

	output = img.copy()
#	print(overlay.shape, output.shape)
#	print(img.shape, output.shape, overlay.shape)
	if overlay.shape == output.shape:
		result = cv2.addWeighted(overlay, 1.0, output, 1.0, 0)
	else:
		result = output

	return result


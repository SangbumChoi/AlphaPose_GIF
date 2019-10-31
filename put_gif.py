from gif_properties import convert_gif_to_frames
import cv2
import numpy as np
import sys
import math
from PIL import Image, ImageDraw, ImageFilter
from PIL import GifImagePlugin
#	np.set_printoptions(threshold=10000)

def put_gif(im_name, img, part_line, position, rotate_start, rotate_end, scale, replay_speed, start_frame, end_frame, file_name):

	im_number = im_name.split('.')[0]
	int_im_number = int(im_number)

	if start_frame <= int_im_number and int_im_number <= end_frame:
		X1 = part_line[position][0]
		Y1 = part_line[position][1]
		rotate_x_start = part_line[rotate_start][0]
		rotate_y_start = part_line[rotate_start][1]
		rotate_x_end = part_line[rotate_end][0]
		rotate_y_end = part_line[rotate_end][1]

		if rotate_x_start == rotate_x_end and rotate_y_start != rotate_y_end:
			angle = -90
		if rotate_x_start == rotate_x_end and rotate_y_start == rotate_y_end:
			angle = 0
		else:
			angle = math.degrees(math.atan2(rotate_y_end - rotate_y_start, rotate_x_end - rotate_x_start))
			angle = 180 - angle

		scale_percent = scale

		imageObject = Image.open("{}".format(file_name))

		(height, width) = img.shape[:2]
	
		portion, gif_number = divmod(int(int_im_number/replay_speed), int(imageObject.n_frames))

		imageObject.seek(gif_number)
		gif = imageObject.copy()

		gif_watermark = gif.convert('RGBA')
		r, g, b, a = gif_watermark.split()
		gif_watermark = Image.merge('RGBA', (b, g, r, a))

		(gif_height, gif_width) = gif_watermark.size

		gif_height1 = int(gif_height * scale_percent / 100)
		gif_width1 = int(gif_width * scale_percent / 100)
		dim = (gif_height1, gif_width1)

		resized_gif_watermark = gif_watermark.resize(dim, Image.ANTIALIAS)
		resized_rotated_gif_watermark = resized_gif_watermark.rotate(angle)
		(resized_gif_height, resized_gif_width) = np.array(resized_gif_watermark).shape[:2]

		start_Y = int(Y1-resized_gif_height/2)
		start_X = int(X1-resized_gif_width/2)

	#	current img format is numpy array but to fit with transparent we must change into PIL format
		img = Image.fromarray(img.astype('uint8'),'RGB')

		transparent = Image.new('RGBA', (width, height), (0,0,0,0))
		transparent.paste(img, (0,0))
		transparent.paste(resized_rotated_gif_watermark, (start_X, start_Y), mask=resized_rotated_gif_watermark)
		result = np.array(transparent)
		result = np.delete(result,[3],2)
	else:
		result = img

	return result

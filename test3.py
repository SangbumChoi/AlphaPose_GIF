from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np

file_name = 'examples/gif/lips.gif'

imageObject = Image.open("{}".format(file_name))

#im_number = im_name.split('.')[0]
try:
	gif_number = 0
	while 1:
		print(imageObject.n_frames)
#		imageObject.show()
#		print(imageObject)
#		print(copy.n_frames)
		imageObject.seek(gif_number)
		copy = imageObject.copy()
#		print(copy)
#		copy.show()
		imageObject.show()
		#	PIL input is BGR and CV2 input is RGB
		A = imageObject.convert('RGBA')
#		imageObject.show()
		#gif_scale2 = np.array(gif_watermark)
		r, g, b, a = A.split()
#		print(r, g, b, a)
		gif_watermark = Image.merge('RGBA', (b, g, r, a))

#		print(gif_watermark)
		#	just for figuring out the scale of watermark
		gif_scale = np.array(gif_watermark)
		(gif_height, gif_width) = gif_scale.shape[:2]

#		print(gif_scale)
		gif_number += 1
except EOFError:
	pass

#cv2.imshow("image",gif_scale)
#cv2.imshow("image2",gif_scale2)
#cv2.waitKey();

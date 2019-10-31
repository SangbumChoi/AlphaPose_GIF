from moviepy.editor import *

clip = (VideoFileClip("examples/res/BetaPose_WIN_20191009_00_33_54_Pro.avi").subclip((0,00.00),(0,03.00)).resize(1.0))
clip.write_gif("test1.gif")

import cv2
import libardrone
from time import time

def main():
	camera = cv2.VideoCapture('tcp://192.168.1.1:5555')
	running, frame = camera.read()
	cv2.imshow('View', frame)
	flying = False
	data_dir = 'data/unclassified/'
	lasttime_write = -1
	while running:
		running, frame = camera.read()
		cv2.imshow('View', frame)
		key = (cv2.waitKey(1)& 0xFF)
		if key == 27:
			running = False
		if key == ord('z'):
			flying = True
		elif key == ord('x'):
			flying = False
		if flying == True:
			imagefile = data_dir + str(int(time())) + '.jpg'
			cv2.imwrite(imagefile, frame)
	camera.release()
	cv2.destroyAllWindows()

            
if __name__ == '__main__':
	main()
from cv2 import cv
import image_process as ip
import cv2
import os
import os.path
import numpy as np
rootdir = "gray/"
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('--datadir', type=str, default=rootdir)

    return parser.parse_args()

def key_index(key):
    idx = -1
    if key == ord('w'):
        idx = 0
    elif key == ord('a'):
        idx = 1
    elif key == ord('d'):
        idx = 2
    elif key == ord('s'):
        idx = 3
    elif key ==  ord('i'):
        idx = 4
    elif key == ord('k'):
        idx = 5
    elif key == ord('j'):
        idx = 6
    elif key == ord('l'):
        idx = 7
    elif key == ord('z'):
        idx = 8
    elif key == ord('x'):
        idx = 9
    elif key == ord('h'):
        idx = 10
    return idx

def main():
    data_shape = (1, 20, 20)
    out = open("data_image",'w+')
    for parent, dirnames, filenames in os.walk(rootdir):
        '''
        for dirname in dirnames:
            print "parent is:" + parent
            print  "dirname is:" + dirname
        '''
        for filename in filenames:
            print "filename is:" + filename
            print "the full name of the file is: " + os.path.join(parent, filename)
            imagefile = os.path.join(parent, filename)
            nn = imagefile.find('.jpg')
            if nn < 0:
                continue
            image = cv2.imread(imagefile, 0)
            image = cv2.resize(image, (data_shape[1], data_shape[2]), interpolation=cv2.INTER_CUBIC)
            image = image.ravel()
            print image
            print image.shape
            #image = cv2.Laplacian(image,cv2.CV_64F)
            #print type(image)
            #print image.shape
            #cv2.imshow('View', image)
            for ele in image:
                out.write(str(ele) + ' ')
            #out.write('\n')
            n0 = imagefile.find('/0/')
            n1 = imagefile.find('/1/')
            n2 = imagefile.find('/2/')
            #n0 = imagefile.index('/0/')
            #n1 = imagefile.index('/1/')
            #n2 = imagefile.index('/2/')

            if n0 >= 0:
                print n0
                imagefile = "edge/0/" + filename
                out.write('0\n')
            elif n1 >= 0:
                print n1
                imagefile = "edge/1/" + filename
                out.write('1\n')
            else:
                print n2
                imagefile = "edge/2/" + filename
                out.write('2\n')
            print imagefile
            #cv2.waitKey(0)
            #cv2.imwrite(imagefile, image)

if __name__ == '__main__':
    args = parse_args()
    print args.datadir
    if not (args.datadir == None):
        rootdir = args.datadir
    main()
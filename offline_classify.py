import cv2
import os
import os.path
rootdir = "data/raw"
todir = 'data/rgb/'
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
    Classify_End = False
    for parent,dirnames,filenames in os.walk(rootdir):
        #print type(filenames)
        list.sort(filenames)
        #print filenames
        for filename in filenames:
            print "filename is:" + filename
            print rootdir + '/' + filename
            image = cv2.imread(rootdir + '/' + filename)
            cv2.imshow("Image", image)
            key = (cv2.waitKey(0) & 0xFF)
            idx = key_index(key)
            path = os.path.join(parent,filename)
            if key == 27:
                break
            if not(idx == -1):
                cv2.imwrite(todir +  str(idx) + '/' + filename, image)
                print 'Save: ' + todir +  str(idx) + '/' + filename
            os.remove(path)
        if Classify_End == True:
            break
cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()
    print args.datadir
    if not (args.datadir == None):
        rootdir = args.datadir
    main()
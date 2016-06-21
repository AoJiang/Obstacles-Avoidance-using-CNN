import os
import cv2
def move(rootdir, todir, count):
    num = 0
    for parent, dirnames, filenames in os.walk(rootdir):
        #filenames.sort()
        for filename in filenames:
            print "filename is:" + filename
            print "the full name of the file is: " + os.path.join(parent, filename)
            imagefile = os.path.join(parent, filename)
            image = cv2.imread(imagefile)
            #cv2.imshow('Image', image)
            #cv2.waitKey(0)
            tofilename = todir + filename
            print 'Tofilename:' + tofilename
            cv2.imwrite(tofilename, image)
            num += 1
            path = os.path.join(parent, filename)
            os.remove(path)
            if num >= count:
                break
        if num >= count:
            break
def delete(rootdir, str):
    for parent, dirnames, filenames in os.walk(rootdir):
        filenames.sort()
        for filename in filenames:
            print "filename is:" + filename
            print "the full name of the file is: " + os.path.join(parent, filename)
            imagefile = os.path.join(parent, filename)
            if imagefile.find(str) >= 0:
                path = os.path.join(parent, filename)
                os.remove(path)

def copy(rootdir, todir, ratio):
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            print "filename is:" + filename
            print "the full name of the file is: " + os.path.join(parent, filename)
            imagefile = os.path.join(parent, filename)
            tofilename = todir + filename
            nratio = 0
            image = cv2.imread(imagefile)
            while nratio < ratio:
                #cv2.imshow('Image', image)
                ratiofile = str(nratio) + 'copy' + filename
                ratiofile = os.path.join(parent, ratiofile)
                nratio += 1
                cv2.imwrite(ratiofile, image)
                print ratiofile
if __name__ == '__main__':
    #move_file('data/rgb_test/2/', 'data/rgb/2/',2500)
    #delete_file('data/rgb/2', 'copy')
    #move('data/rgb_tr/0/', 'data/rgb_te/0/', 3500)
    copy('data/rgb_te/2/','data/rgb_te/2/', 10)
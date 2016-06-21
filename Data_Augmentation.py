import random
import cv2
import os
import image_process as ip
import numpy as np
def generate_PCA(rootdir, todir, ratio):
    pixel = None
    count = 0
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            print "filename is:" + filename
            print "the full name of the file is: " + os.path.join(parent, filename)
            imagefile = os.path.join(parent, filename)
            #print imagefile
            nn = imagefile.find('.jpg')
            if nn < 0:
                continue
            image = cv2.imread(imagefile)
            if random.random() > 0.5:
                new_image = np.reshape(image, (230400, 3))
                new_image = np.multiply(new_image, 1.0 / 255)
                arr = range(10)
                random.shuffle(arr)
                #print new_image.shape
                if pixel == None:
                    pixel = new_image[arr,:]
                else:
                    print '!'
                    print new_image[arr,:]
                    print '!!'
                    print (new_image[arr,:].shape)
                    pixel = np.vstack((pixel, new_image[arr,:]))
                #print pixel
                print pixel.shape
                count += 1
                if count == 100:
                    break
            #cv2.waitKey(0)
        if count == 100:
            break
    cov_mat = np.cov(pixel)
    print cov_mat
    eigval, eigvector = np.linalg.eig(cov_mat)
    print 'EigVal:'
    print eigval
    alpha = [random.gauss(0, 0.1) for i in [0, 1, 2]]
    print 'Alpha:'
    print alpha
    eigval = eigval + alpha
    print 'EigVal:'
    print eigval
    print 'EigVector:'
    print eigvector
    print 'Add:'
    add = np.dot(eigval, eigvector)
    print add
    print add.shape

    # for parent, dirnames, filenames in os.walk(rootdir):
    #     for filename in filenames:
    #         print "filename is:" + filename
    #         print "the full name of the file is: " + os.path.join(parent, filename)
    #         imagefile = os.path.join(parent, filename)
    #         print imagefile
    #         nn = imagefile.find('.jpg')
    #         if nn < 0:
    #             continue
    #         pic = cv2.imread(imagefile)
    #         pic = np.reshape(pic, (230400, 3))
    #         pic += add
    #         pic = np.reshape(pic, (360, 640, 3))
    #         cv2.imwrite()

def PCA_Augmentation():
    #cv2.imshow('Old', nimage)
    sizex = nimage.shape[1]
    sizey = nimage.shape[0]
    b, g, r = cv2.split(nimage)
    print b
    cv2.imshow("Blue", r)
    cv2.imshow("Red", g)
    cv2.imshow("Green", b)
    chan = []
    chan.append(b.ravel())
    chan.append(g.ravel())
    chan.append(r.ravel())
    chan = np.array(chan)
    cov_mat = np.cov(chan)
    print cov_mat
    #print cov_mat
    eigval, eigvector = np.linalg.eig(cov_mat)
    alpha = [random.gauss(0,0.1) for i in [0,1,2]]
    eigval = eigval + alpha
    #print eigval
    add =  np.dot(eigval,eigvector)
    print b.shape
    print 'Add'
    print add
    print add[0] * np.ones([sizey, sizex])
    print 'Add'
    b = b + alpha[0] * add[0] * np.ones([sizey, sizex])
    g = g + alpha[1] * add[1] * np.ones([sizey, sizex])
    r = r + alpha[2] * add[1] * np.ones([sizey, sizex])
    print b
    new_image = cv2.merge([b,g,r])
    print new_image.shape
    cv2.imshow('Image',new_image)
    cv2.waitKey(0)
    return new_image

if __name__ == '__main__':
    # m = cv2.imread('1464514231.jpg')
    # a = np.array([[1, 2, 3], [4, 5, 6]])
    # b = np.array([0, 1, 2])
    # c = np.vstack((a,b))
    # print b
    # random.shuffle(b)
    # print b
    # #print random.shuffle(range(0))
    # print c
    # print c[b, :]
    #
    # print m
    # print '!!!'
    # m = np.reshape(m, (230400, 3))
    # print '!!!'
    # print m
    # print '!!!'
    # print m + b
    # #b = np.array([[1, 2, 3], [4, 5, 6]])
    #c = np.hstack((a,b))
    #print c
    generate_PCA('data/rgb/', None, 0)
    #a = np.array([1,2,3])
    #b = np.array([1,2,3])
    #print a + b

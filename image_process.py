import cv2.cv as cv
import cv2
import os
import random

from sklearn.decomposition import PCA
import numpy as np
alpha = [-0.0662134244800563, 0.09762089480206204, -0.03597601509149528]
def random_array(total_num, num):
    array = range(total_num)
    random.shuffle(array)
    array = array[:num]
    array.sort()
    return array

def random_crop(image, data_shape):
    orix = image.shape[1]
    oriy = image.shape[0]
    newx = data_shape[1]
    newy = data_shape[0]
    randy = random_array(oriy, newy)
    randx = random_array(orix, newx)
    new_image = image[randy, :]
    new_image = new_image[:, randx]
    return new_image


def generate_randcrop(rootdir, todir, data_shape, ratio):
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            print "filename is:" + filename
            print "the full name of the file is: " + os.path.join(parent, filename)
            imagefile = os.path.join(parent, filename)
            nn = imagefile.find('.jpg')
            n0 = imagefile.find('/0/')
            n1 = imagefile.find('/1/')
            n2 = imagefile.find('/2/')
            if nn < 0:
                continue
            image = cv2.imread(imagefile, 0)
            if n0 >= 0:
                imagefile = todir + "0/" + filename[:filename.find('.jpg')]
            elif n1 >= 0:
                imagefile = todir + "1/" + filename[:filename.find('.jpg')]
            else:
                imagefile = todir + "2/" + filename[:filename.find('.jpg')]
            print imagefile
            nratio = 0
            while nratio < ratio:
                new_image = random_crop(image, data_shape)
                #cv2.imshow("New", new_image)
                #cv2.waitKey(0)
                ratiofile = imagefile + str(nratio) + 'crop.jpg'
                nratio += 1
                #cv2.waitKey(0)
                cv2.imwrite(ratiofile, new_image)

def PCA_Augmentation(nimage):
    alpha = [-0.0662134244800563, 0.09762089480206204, -0.03597601509149528]
    #cv2.imshow('Old', nimage)
    sizex = nimage.shape[1]
    sizey = nimage.shape[0]
    B, G, R = cv2.split(nimage)
    y = [random.randint(1, sizey), random.randint(1, sizey)]
    y.sort()
    x = [random.randint(1, sizex), random.randint(1, sizex)]
    x.sort()
    patch = (y,x)
    maxB = np.max(B)
    maxG = np.max(G)
    maxR = np.max(R)
    #HSV_noise(newH, patch, 'add', add = maxH)
    patch = ([0,359], [0, 639])
    b = HSV_noise(B, patch, 'add', maxB, add = alpha[0])
    g = HSV_noise(G, patch, 'add', maxG, add = alpha[1])
    r = HSV_noise(R, patch, 'add', maxR, add = alpha[2])
    # print b
    # cv2.imshow("Blue", r)
    # cv2.imshow("Red", g)
    # cv2.imshow("Green", b)
    # chan = []
    # chan.append(b.ravel())
    # chan.append(g.ravel())
    # chan.append(r.ravel())
    # chan = np.array(chan)
    # cov_mat = np.cov(chan)
    # print cov_mat
    # #print cov_mat
    # eigval, eigvector = np.linalg.eig(cov_mat)
    # alpha = [random.gauss(0,0.1) for i in [0,1,2]]
    # eigval = eigval + alpha
    # #print eigval
    # add =  np.dot(eigval,eigvector)
    # print b.shape
    # print 'Add'
    # print add
    # print add[0] * np.ones([sizey, sizex])
    # print 'Add'
    # b = b + alpha[0] * add[0] * np.ones([sizey, sizex])
    # g = g + alpha[1] * add[1] * np.ones([sizey, sizex])
    # r = r + alpha[2] * add[1] * np.ones([sizey, sizex])
    # print b
    new_image = cv2.merge([b,g,r])
    print new_image.shape
    cv2.imshow('Image',new_image)
    #cv2.waitKey(0)
    return new_image

def generate_pca(rootdir, todir, ratio):
    vr = vg = vb = []
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            print "filename is:" + filename
            print "the full name of the file is: " + os.path.join(parent, filename)
            imagefile = os.path.join(parent, filename)
            image = cv2.imread(imagefile)
            image = np.reshape(image, (230400, 3))



def generate_PCA(rootdir, todir, ratio):
    pr = []
    pb = []
    pg = []
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            print "filename is:" + filename
            print "the full name of the file is: " + os.path.join(parent, filename)
            imagefile = os.path.join(parent, filename)
            nn = imagefile.find('.jpg')
            n0 = imagefile.find('/0/')
            n1 = imagefile.find('/1/')
            n2 = imagefile.find('/2/')
            if nn < 0:
                continue
            image = cv2.imread(imagefile)
            if n0 >= 0:
                imagefile = todir + "0/" + filename[:filename.find('.jpg')]
            elif n1 >= 0:
                imagefile = todir + "1/" + filename[:filename.find('.jpg')]
            else:
                imagefile = todir + "2/" + filename[:filename.find('.jpg')]
            print imagefile
            nratio = 0
            while nratio < ratio:
                #cv2.imshow('Image',image)
                new_image = PCA_Augmentation(image)
                #cv2.imshow('Image2', image)
                #cv2.imshow("New", new_image)
                #cv2.waitKey(0)
                ratiofile = imagefile + str(nratio) + 'pca.jpg'
                nratio += 1
                #cv2.waitKey(0)
                cv2.imwrite(ratiofile, new_image)

def compute_mean(rootdir):
    mr = []
    mb = []
    mg = []
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            print "filename is:" + filename
            print "the full name of the file is: " + os.path.join(parent, filename)
            imagefile = os.path.join(parent, filename)
            nn = imagefile.find('.jpg')
            if nn < 0:
                continue
            image = cv2.imread(imagefile)
            b, g, r = cv2.split(image)
            cv2.imshow('Image', image)
            #tran = np.reshape(image, (3, 360, 640))
            mr.append(np.mean(r))
            mg.append(np.mean(g))
            mb.append(np.mean(b))
            #cv2.waitKey(0)
    print np.mean(mr)
    print np.mean(mg)
    print np.mean(mb)

def HSV_noise(npic, area, op, max_value, add = None):
    print max_value
    pic = npic.astype(np.float)
    pic = np.multiply(pic, 1.0 / max_value)
    print 'Area:' + str(area)
    idx = 0;idy = 0
    sizex = pic.shape[1]
    sizey = pic.shape[0]
    if op == 'add':
        if add == None:
            add = random.uniform(0, 0.1)
        for idy in range(sizey):
            for idx in range(sizex):
                if idy + 1 >= area[0][0] and idy + 1<= area[0][1] and idx + 1 >= area[1][0] and idx + 1 <= area[1][1]:
                    pic[idy,idx] += add
                    pic[idy,idx] = max(pic[idy,idx], 0)
                    pic[idy,idx] = min(pic[idy,idx], 1)
    elif op == 'power':
        power = random.uniform(0.25, 4)
        for idy in range(sizey):
            for idx in range(sizex):
                if idy + 1 >= area[0][0] and idy + 1 <= area[0][1] and idx + 1 >= area[1][0] and idx + 1 <= area[1][1]:
                    pic[idy, idx] = pow(pic[idy, idx], power)
                    pic[idy, idx] = max(pic[idy, idx], 0)
                    pic[idy, idx] = min(pic[idy, idx], 1)
    elif op =='multiply':
        mul = random.uniform(0.7, 1.4)
        for idy in range(sizey):
            for idx in range(sizex):
                if idy + 1 >= area[0][0] and idy + 1 <= area[0][1] and idx + 1 >= area[1][0] and idx + 1 <= area[1][1]:
                    pic[idy, idx] *= mul
                    pic[idy, idx] = max(pic[idy, idx], 0)
                    pic[idy, idx] = min(pic[idy, idx], 1)
    pic = np.multiply(pic, max_value)
    pic = pic.astype(np.uint8)
    #print pic
    return pic

def HSV_Augmentation(image):
    cv2.imshow('Old', image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sizex = image.shape[1]
    sizey = image.shape[0]
    H, S, V = cv2.split(image)
    y = [random.randint(1, sizey), random.randint(1, sizey)]
    y.sort()
    x = [random.randint(1, sizex), random.randint(1, sizex)]
    x.sort()
    patch = (y,x)
    maxH = np.max(H)
    maxS = np.max(S)
    maxV = np.max(V)
    newH = H
    newS = S
    newV = V
    if random.random() >= 0.5:
        newH = HSV_noise(newH, patch, 'add', maxH)
    if random.random() >= 0.5:
        newS = HSV_noise(newS, patch, 'power', maxS)
    if random.random() >= 0.5:
        newS = HSV_noise(newS, patch, 'multiply', maxS)
    if random.random() >= 0.5:
        newS = HSV_noise(newS, patch, 'add', maxS)
    if random.random() >= 0.5:
        newV = HSV_noise(newV, patch, 'power', maxV)
    if random.random() >= 0.5:
        newV = HSV_noise(newV, patch, 'multiply', maxV)
    if random.random() >= 0.5:
        newV = HSV_noise(newV, patch, 'add', maxV)
    image = cv2.merge([newH, newS, newV])
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imshow('New', image)
    #cv2.waitKey(0)
    return image

def data_augmentation(method, rootdir, todir, ratio):

    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            print "filename is:" + filename
            print "the full name of the file is: " + os.path.join(parent, filename)
            imagefile = os.path.join(parent, filename)
            nn = imagefile.find('.jpg')
            n0 = imagefile.find('/0/')
            n1 = imagefile.find('/1/')
            n2 = imagefile.find('/2/')
            if nn < 0:
                continue
            image = cv2.imread(imagefile)
            if n0 >= 0:
                imagefile = todir + "0/" + filename[:filename.find('.jpg')]
            elif n1 >= 0:
                imagefile = todir + "1/" + filename[:filename.find('.jpg')]
            else:
                imagefile = todir + "2/" + filename[:filename.find('.jpg')]
            print imagefile
            nratio = 0
            while nratio < ratio:
                cv2.imshow('Image',image)
                ratiofile = imagefile + str(nratio)
                if method == 'pca':
                    new_image = PCA_Augmentation(image)
                    ratiofile += 'pca.jpg'
                elif method == 'hsv':
                    new_image = HSV_Augmentation(image)
                    ratiofile += 'hsv.jpg'
                #cv2.imshow('Image2', image)
                cv2.imshow("New2", new_image)
                nratio += 1
                #cv2.waitKey(0)
                cv2.imwrite(ratiofile, new_image)

if __name__ == '__main__':
    #image = cv2.imread('1464514231.jpg')
    #PCA_Augmentation(image)
    #generate_randcrop('data/rgb/','data/randcrop/',(320,480),20)
    #generate_PCA('data/rgb/', 'data/pca/', 5)
    #compute_mean('data/hsv/')
    #HSV_Augmentation(image)
    data_augmentation('pca','data/rgb/2/', 'data/pca_all/', 10)
    #[-0.0662134244800563, 0.09762089480206204, -0.03597601509149528]
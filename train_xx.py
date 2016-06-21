import find_mxnet
import mxnet as mx
import argparse
import os, sys
import train_network
import numpy as np
import cv2
import network
import image_process as ip
import symbol_resnet
import symbol_vgg
import symbol_inception
def parse_args():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('--network', type=str, default='lenet', choices = ['mlp', 'lenet','alexnet','googlenet','vgg','resnet','mynet','dqn','depNet'], help = 'neural network')
    parser.add_argument('--trainfile', type=str, default='train.rec', help = 'the file used to train')
    parser.add_argument('--data-dir', type=str, default='pca_all', help='the input data directory')
    parser.add_argument('--gpus', type=str, help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--num-examples', type=int, default=3, help='the number of training examples')
    parser.add_argument('--batch-size', type=int, default=128, help='the batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
    parser.add_argument('--model-prefix', default='model/dqn/model', type=str, help='the prefix of the model to load/save')
    parser.add_argument('--save-model-prefix', type=str, default = 'model/dqn/model', help='the prefix of the model to save')
    parser.add_argument('--num-epochs', type=int, default=20, help='the number of training epochs')
    parser.add_argument('--load-epoch', type=int, help="load the model on an epoch using the model-prefix")
    parser.add_argument('--begin-epoch', type=int, default=0, help="the epoch to start training")
    parser.add_argument('--kv-store', type=str, default='local', help='the kvstore type')
    parser.add_argument('--lr-factor', type=float, default=1, help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--lr-factor-epoch', type=float, default=1, help='the number of epoch to factor the lr, could be .5')
    parser.add_argument('--clip_gradient', type=float, default=1.8, help='clip_gradient')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    data_shape = (3, 84, 84)
    if args.network == 'mlp':
        net = network.get_mlp()
        data_shape = (data_shape[1]*data_shape[2],)
    elif args.network == 'alexnet':
        net = network.get_alexnet(3)
    elif args.network == 'lenet':
        net = network.get_lenet(3)
    elif args.network == 'googlenet':
        net = symbol_inception.get_symbol(3)
    elif args.network == 'mynet':
        net = network.get_mynet(3)
    elif args.network == 'vgg':
        net = symbol_vgg.get_symbol(3)
    elif args.network == 'dqn':
        net = network.get_dpn(3)
    elif args.network == 'depNet':
        net = network.get_depNet(3)

    #elif args.network ==

    #net = network.get_alexnet()

    #net = network.get_alexnet(3)
    # train

    #data_shape =
    model = train_network.fit(args, net, data_shape)
    pic = cv2.imread('1464514231.jpg')
    print pic.shape
    print type(pic)
    #pic = cv2.imread('gray/1/1464235565.jpg', 1)
    #print pic.shape
    pic = cv2.resize(pic,(data_shape[1],data_shape[2]),interpolation=cv2.INTER_CUBIC)
    #cv2.imshow('View', pic)
    #cv2.waitKey(0)
    print type(pic)
    print pic.shape
    image = np.reshape(pic, (1, data_shape[0], data_shape[1], data_shape[2]))
    print image.shape
    #print type(image)
    print type(image)
    print image.shape
    #image = ip.gray_image(pic, "fenliang")
    #print image.shape
    #print type(image)
    #print image
    test_data = mx.io.NDArrayIter(data = image)
    print test_data.provide_data
    ans = model.predict(test_data)
    print type(ans)
    print ans
    #ans = model.predict(pic)
    #print ans

#!/usr/bin/env python

# Copyright (c) 2011 Bastian Venthur
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


"""Demo app for the AR.Drone.

This simple application allows to control the drone and see the drone's video
stream.
"""


#import pygame
import libardrone
import cv2
import argparse
import network
import train_network
from time import sleep
from time import clock
from time import time
def parse_args():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('--network', type=str, default='lenet', choices = ['mlp', 'lenet'], help = 'the cnn to use')
    parser.add_argument('--trainfile', type=str, default='train.rec', help = 'the file used to train')
    parser.add_argument('--data-dir', type=str, default='data/', help='the input data directory')
    parser.add_argument('--gpus', type=str, help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--num-examples', type=int, default=3, help='the number of training examples')
    parser.add_argument('--batch-size', type=int, default=1, help='the batch size')
    parser.add_argument('--lr', type=float, default=.1, help='the initial learning rate')
    parser.add_argument('--model-prefix', default='model/lenet/model', type=str, help='the prefix of the model to load/save')
    parser.add_argument('--save-model-prefix', type=str, default = 'model/lenet/model', help='the prefix of the model to save')
    parser.add_argument('--num-epochs', type=int, default=10, help='the number of training epochs')
    parser.add_argument('--load-epoch', default=10, type=int, help="load the model on an epoch using the model-prefix")
    parser.add_argument('--begin-epoch', type=int, default=0, help="the epoch to start training")
    parser.add_argument('--kv-store', type=str, default='local', help='the kvstore type')
    parser.add_argument('--lr-factor', type=float, default=1, help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--lr-factor-epoch', type=float, default=1, help='the number of epoch to factor the lr, could be .5')
    return parser.parse_args()

def Neural_Network_Initial(args, data_shape, trainfile):
    print "Initialling..."
    net = network.get_lenet()
    return  train_network.fit(args, net, data_shape, 10)

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

def index_key(idx):
    if idx == 0:
        return  ord('w')
    elif idx == 1:
        return  ord('a')
    elif idx == 2:
        return ord('d')
    elif idx == 3:
        return  ord('s')
    elif idx == 4:
        return ord('i')
    elif idx == 5:
        return  ord('k')
    elif idx == 6:
        return  ord('j')
    elif idx == 7:
        return  ord('l')
    elif idx == 8:
        return  ord('z')
    elif idx == 9:
        return ord('x')
        idx = 9
    elif idx == 10:
        return ord('h')
def key_valide(key):
    if key == ord('z') or key == ord('x') or key == ord('h') or key == ord('y') or key == ord('w') or key == ord('s') or key == ord('a') or key == ord('d') or key == 27:
        return True
    elif key == ord('n') or key == ord('m'):
        return True
    return False
def main(model, dat):
    print "Running Main..."
    camera = cv2.VideoCapture('tcp://192.168.1.1:5555')
    running, frame = camera.read()
    #cv2.imshow('View', frame)
    drone = libardrone.ARDrone()
    drone.speed = 0.05
    data_dir = "data/unclassified/"
    count = 1.5
    Flying = False
    control_state = 0
    while running:
        running, frame = camera.read()
        cv2.imshow('View2', frame)
        key = (cv2.waitKey(1) & 0xFF)
        print "Running..."
        imagefile = data_dir + str(int(time()))
        cv2.imwrite(imagefile + '.jpg', frame)
        if key_valide(key) == True:
            print "Key Press is " + chr(key)
        if key_valide(key):
            if key == 27:
                running = False
            elif key == ord('n'):
                control_state = 0
            elif key == ord('m'):
                control_state = 1;
            elif key == ord('z'):
                drone.takeoff()
                drone.speed=0.05
                flying = True
            elif key == ord('x'):
                drone.land()
                drone.speed=0
                flying = False
            elif key == ord('h'):
                drone.hover()
            elif key == ord('y'):
                drone.reset()
            if Flying == True:
                continue
            if control_state == 0:
                print "Manual"
                if key == ord('w'):
                    Flying = True
                    print "Forward"
                    drone.move_forward()
                    sleep(count)
                    drone.hover()
                elif key == ord('s'):
                    Flying = True
                    print "Back"
                    drone.move_backward()
                    sleep(count)
                    drone.hover()
                elif key == ord('a'):
                    Flying = True
                    print "Left"
                    drone.move_left()
                    sleep(count)
                    drone.hover()
                elif key == ord('d'):
                    Flying = True
                    print "Right"
                    drone.move_right()
                    sleep(count)
                    drone.hover()
                else:
                    print "Manual Not Valid"
                Flying = False
            elif control_state == 1:
                print "Auto"
                ans = train_network.predict(model=model, image=frame, data_shape = data_shape)
                print ans
                key = index_key(ans)
                print key
                if key == ord('w'):
                    Flying = True
                    print "Forward"
                    drone.move_forward()
                    sleep(count)
                    drone.hover()
                elif key == ord('s'):
                    Flying = True
                    print "Back"
                    drone.move_backward()
                    sleep(count)
                    drone.hover()
                elif key == ord('a'):
                    Flying = True
                    print "Left"
                    drone.move_left()
                    sleep(count)
                    drone.hover()
                elif key == ord('d'):
                    Flying = True
                    print "Right"
                    drone.move_right()
                    sleep(count)
                    drone.hover()
                else:
                    print "Auto Not Valid"
                Flying = False
            '''
            elif key ==  ord('i'):
                drone.move_up()
            elif key == ord('k'):
                drone.move_down()
            elif key == ord('j'):
                drone.turn_left()
            elif key == ord('l'):
                drone.turn_right()
            '''
        else:
            # error reading frame
        	print 'error reading video feed'
    flying = False
    print "Shutting down..."
    drone.halt()
    print "Ok."
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    data_dir = "data/unclassified/"
    data_shape = (3,45, 80)
    args = parse_args()
    args.begin_epoch = args.load_epoch
    model = Neural_Network_Initial(args = args, data_shape = data_shape, trainfile = args.trainfile)
    main(model, data_shape)

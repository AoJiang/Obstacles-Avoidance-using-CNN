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
import ps_drone as ps
import cv2
import argparse
import network
import train_network
from time import sleep
from time import clock
from time import time
import threading

#mylock = threading.RLock()

global on_the_air
global running
def parse_args():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('--network', type=str, default='lenet', choices = ['mlp', 'lenet'], help = 'the cnn to use')
    parser.add_argument('--trainfile', type=str, default='gray.rec', help = 'the file used to train')
    parser.add_argument('--data-dir', type=str, default='gray/', help='the input data directory')
    parser.add_argument('--gpus', type=str, help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--num-examples', type=int, default=3, help='the number of training examples')
    parser.add_argument('--batch-size', type=int, default=128, help='the batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='the initial learning rate')
    parser.add_argument('--model-prefix', default='model/lenet/model', type=str, help='the prefix of the model to load/save')
    parser.add_argument('--save-model-prefix', type=str, default = 'model/lenet/model', help='the prefix of the model to save')
    parser.add_argument('--num-epochs', type=int, default=20, help='the number of training epochs')
    parser.add_argument('--load-epoch', type=int, default=20,help="load the model on an epoch using the model-prefix")
    parser.add_argument('--begin-epoch', type=int, default=20, help="the epoch to start training")
    parser.add_argument('--kv-store', type=str, default='local', help='the kvstore type')
    parser.add_argument('--lr-factor', type=float, default=1, help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--lr-factor-epoch', type=float, default=1, help='the number of epoch to factor the lar, could be .5')
    parser.add_argument('--image-collect', type=bool, default=False, choices=[True, False], help='whether to collect image')
    parser.add_argument('--collect-dir', type=str, default='data/raw/', help='the dir to save image')
    parser.add_argument('--control-mode', type=str, default='Manual', choices=['Auto','Manual'], help='the dir to save image')
    return parser.parse_args()


def Neural_Network_Initial(args, data_shape, trainfile):
    print "Initialling..."
    net = network.get_lenet()
    return  train_network.fit(args, net, data_shape)

def strindex_key(idx):
    if idx == 0:
        return  'w'
    elif idx == 1:
        return  'a'
    elif idx == 2:
        return 'd'
    elif idx == 3:
        return  's'
    elif idx == 4:
        return 'i'
    elif idx == 5:
        return  'k'
    elif idx == 6:
        return  'j'
    elif idx == 7:
        return  'l'
    elif idx == 8:
        return  'z'
    elif idx == 9:
        return 'x'
    elif idx == 10:
        return ord('h')
def strkey_valide(key):
    if key in ['n','m','w','a','s','d','z','x','b','y','h','i','k','l','j']:
        return True
    return  False
def control(drone,now_key,count):
    if now_key == 'w':
        print "Forward"
        drone.moveForward()
    elif now_key == 's':
        print "Back"
        drone.moveBackward()
    elif now_key == 'a':
        print "Left"
        drone.moveLeft()
    elif now_key == 'd':
        print "Right"
        drone.moveRight()
    elif now_key == 'i':
        print "Up"
        drone.moveUp()
    elif now_key == 'k':
        print "Down"
        drone.moveDown()
    elif now_key == 'j':
    	print 'Turn Left'
    	drone.turnLeft()
    elif now_key == 'l':
    	print 'Turn Right'
    	drone.turnRight()
    sleep(count)
    drone.stop()
def grayscale(frame, data_shape):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return image

def image_collect():
    global on_the_air
    global running
    while True:
        #print 'On the Air:' + str(on_the_air)
        sleep(0.05)
        if on_the_air == True:
            filename = args.collect_dir + str(int(time()*1000)) + '.jpg'
            print filename
            cv2.imwrite(filename, drone.VideoImage)
        if running == False:
            break
def ardrone_control(model = None, data_shape = None):
    global on_the_air
    print "Running Main..."
    data_dir = args.image_collect
    count = 2.0
    control_state = 0
    on_the_air = False
    running = True
    while running:
        now_key = drone.getKey()
        if strkey_valide(now_key):
            print "Key Press is " + now_key
            if now_key == 'b':
                running = False
                control_state = 0
                break
            elif now_key == 'n':
                control_state = 0
            elif now_key == 'm':
                if args.control_mode == 'Manual':
                    continue
                control_state = 1
            elif now_key == 'z':
                drone.takeoff()
                drone.setSpeed(0.05)
                on_the_air = True
            elif now_key == 'x':
                drone.setSpeed(0.0)
                drone.land()
                on_the_air = False
                control_state = 0
            elif now_key == 'h':
                drone.stop()
                control_state = 0
            elif now_key == 'y':
                drone.reset()
        if control_state == 0 and strkey_valide(now_key):
            print "Manual"
            control(drone,now_key,count)
        elif control_state == 1:
            now_frame = drone.VideoImage
            print "Auto"
            now_frame = cv2.cvtColor(now_frame, cv2.COLOR_BGR2GRAY)
            ans = train_network.predict(model=model, image=now_frame, data_shape = data_shape)
            print ans
            now_key = strindex_key(ans)
            print now_key
            control(drone, now_key, count)
            sleep(5)
if __name__ == '__main__':
    data_dir = "data/raw/"
    data_shape = (1,100, 100)
    args = parse_args()
    args.begin_epoch = args.load_epoch
    model = False
    if args.control_mode == 'Auto':
        model = Neural_Network_Initial(args = args, data_shape = data_shape, trainfile = args.trainfile)
    drone = ps.Drone()
    drone.startup()
    drone.reset()
    while (drone.getBattery()[0] == -1): sleep(0.1)
    print "Battery: " + str(drone.getBattery()[0]) + "% " + str(drone.getBattery()[1])
    drone.useDemoMode(True)

    drone.setConfigAllID()
    drone.sdVideo()
    drone.frontCam()
    CDC = drone.ConfigDataCount
    while CDC == drone.ConfigDataCount: sleep(0.001)
    drone.startVideo()
    drone.showVideo()

    #drone.setSpeed(0.0)
    #drone.stop()
    global on_the_air
    global running
    on_the_air = False
    running = True
    while drone.VideoImage == None:
        print "Ardrone is Waitting"
        sleep(5)
    #drone.saveVideo(True)
    t2 = threading.Thread(target=image_collect)
    if args.image_collect == True:
        t2.setDaemon(True)
        t2.start()
    #t1 = threading.Thread(target=ardrone_control(model,data_shape),args=(model, data_shape, ))
    #t1.start()
    ardrone_control(model = model, data_shape = data_shape)
    print "End"
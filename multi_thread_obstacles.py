import threading
from time import time,sleep
import cv2

global Running
global on_the_air
#global data_dir
global key
global frame
#global camer
def key_valide(key):
    if key == ord('z') or key == ord('x') or key == ord('h') or key == ord('y') or key == ord('w') or key == ord('s') or key == ord('a') or key == ord('d') or key == 27:
        return True
    elif key == ord('n') or key == ord('m'):
        return True
    return False
def show_image():
    print "Show_Image"
    global frame
    global Running
    #global camer
    camera = cv2.VideoCapture('tcp://192.168.1.1:5555')
    Running, frame = camera.read()
    while Running == True:
        print("Show_Image is Running...")
        Running, frame = camera.read()
        cv2.imshow("Show_Image", frame)
        key = (cv2.waitKey(10) & 0xFF)
        #print "On the air " + str(on_the_air)
        #if on_the_air == True:
        #    imagefile = data_dir + str(int(time()))
        #    cv2.imwrite(imagefile + '.jpg', frame)
        if key_valide(key) == True:
            print "Key Press is " + chr(key)
            if key == 27:
                Running = False
    print "Show_Image End"
    cv2.destroyWindow("Show_Image")
    #threading.
def ardrone_control():
    global Running
    global frame
    #for i in range(10):
    #    print "Ardrone_Control"
    while Running == False:
        print "Ardrone is Waitting..."
        sleep(2)
    while Running == True:
        print("Ardrone_Control is Running...")

        #cv2.imshow("Ardrone", frame)
        #cv2.waitKey(0)
        sleep(2)
        #cv2.destroyWindow("Ardrone")
        #cv2.imshow("Ardrone",frame)
        #key2 = (cv2.waitKey(1) & 0xFF)
    #cv2.destroyWindow("Ardrone")
    print "Ardrone_Control End"


if __name__ == '__main__':

    #show_image()
    #ardrone_control()
    global Running
    Running = False
    threads = []
    t1 = threading.Thread(target=show_image)
    threads.append(t1)
    t2 = threading.Thread(target=ardrone_control)
    threads.append(t2)
    #threads[1].setDaemon(True)
    t1.start()
    t2.start()
    #for t in threads:
    #    t.start()
    #for t in threads:
    #    t.join()

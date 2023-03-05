# -*- coding: utf-8 -*-
r"""
Elation Sports Technologies LLC
www.elationsportstechnologies.com

The purpose of this script is to collect data from the sensors on the rebounder,
as well as the webcam, tracking a green spray-painted soccer ball, in order
to create a dataset for training a neural network, to relate the landing position
and of the ball to accelerometer data.

Installation of libraries:
pip install opencv-python
pip install imutils

"""

from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2 as cv
import imutils
import time, csv

import matplotlib.pyplot as plt

import serial, io, datetime
from serial import Serial

def readData():
    buffer = ""
    while True:
        oneByte = ser.read(1)
        if oneByte == b"\n":
            return buffer
        else:
            buffer += oneByte.decode("ascii")

plt.close('all')

timestr = time.strftime('%d%b%Y_%I%M%p')

folder_path = r'C:\Your Folder File Path Here'

plotAlpha = 0.25

#Include the timestamps in the data
include_timestamp_boolean = True

#Show the serial data in the Python console
show_serial_data_boolean = True

print()
print('Initializing ESP32...')

file_prefix = r'Log' 
file_type = '.csv'
file_name = file_prefix + '_' + timestr + file_type
file_path = folder_path + '\\' + file_name

addr = "COM7" #Serial port to read the data
baud = 9600 

ser = serial.Serial(
    port = addr,\
    baudrate = baud,\
    parity=serial.PARITY_NONE,\
    stopbits=serial.STOPBITS_ONE,\
    bytesize=serial.EIGHTBITS,\
    timeout=0)

#Test that the data can be read
print()
print('Attempting to read 1 x line from the ESP32 output...')
print("If we arent proceeding past this line, check the baud rate.")
print("If that doesnt work, unplug (then run this code while unplugged) and then replug the ESP32 and re-run the code.")
line = readData()

print('Initial test readData(): ' + str(line))
if line[0:4] == 'Fail':
    raise('MPU6050 issue!')

print('Success - ESP32 test line read!')

print()
print('Initializing camera...')

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

greenLower = np.array([70, 40, 40])   
greenUpper = np.array([120, 255, 200])


pts = deque(maxlen=args["buffer"])

print()
print('Starting camera stream, please wait 10 seconds...')
print('If the camera stream doesnt appear, try plugging it into a different USB port.')

if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    raise('Pause here')
time.sleep(2.0)

videoCapture = vs
prevCircle = None #Current circle

#Define distance function to calculate the distance between two points in the image
dist = lambda x1,y1,x2,y2: ((x2-x1)**2+(y2-y1)**2)*0.5

print()
print('Success - camera initialized!')
print()

seq = []
count = 1
start_time = time.time()
try:
    with open(file_path, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        while True:
            time_curr = time.time() - start_time
            #Sensor stuff - the ESP32  will spit out the accelerometer data
            #one sensor after the other, so 4 x reads will collect 4 x datapoints
            #from each of the sensors.
            
            #Take readings and discard them until the final sensor is read, then proceed to read the next line.
            #This should work out such that the next sensor reading is reading #1.
            proceed_bool = False
            while proceed_bool == False:
                line = readData()
                #print(line)
                if line[0] == '4':
                    proceed_bool = True
            
            for i in range(0,4):
                line = readData()
                
                if show_serial_data_boolean:
                    print(line)
                
                if include_timestamp_boolean != True:
                    to_write = [str(line)]
                else:
                    to_write = [str(time_curr) + ',' + str(line)]
                
                #Remove trailing carriage return
                to_write[0] = to_write[0][:-1]
                
                spamwriter.writerow(to_write)
            
            frame = videoCapture.read()
            frame = imutils.resize(frame, width=600)
            frame = cv.flip(frame, 0) #Flip the image over vertically
            blurred = cv.GaussianBlur(frame, (5, 5), 0)
            hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
            
            mask = cv.inRange(hsv, greenLower, greenUpper)
            mask = cv.erode(mask, None, iterations=2)
            mask = cv.dilate(mask, None, iterations=2)
        
            cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            center = None
            if len(cnts) > 0:
                c = max(cnts, key=cv.contourArea)
                ((x, y), radius) = cv.minEnclosingCircle(c)
                M = cv.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if radius > 10:
                    cv.circle(frame, (int(x), int(y)), int(radius),
                        (0, 255, 255), 2)
                    cv.circle(frame, center, 5, (0, 0, 255), -1)
            else:
                ((x, y), radius) = ((0,0), 0)
            pts.appendleft(center)
            
            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue
                thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
                cv.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
            
            cv.imshow("Frame", frame)
            
            spamwriter.writerow([str(time_curr) + ',' + '5' + ',' + str(x) + ',' + str(y) + ',' + str(radius)])
            
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
            
    
except Exception as e:
    print(e)
    print('Exception occured.')
    
    print()
    print('Log data saved to:')
    print(file_name)
    
    ser.close()

if not args.get("video", False):
	vs.stop()
else:
	vs.release()
cv.destroyAllWindows()





# -*- coding: utf-8 -*-
r"""
Elation Sports Technologies LLC
https://www.elationsportstechnologies.com/

Sensing Rebounder Project with Accelerometers
Real-Time Neural Network Inference Script

pip install tensorflow
pip install pyserial


"""

import numpy as np
import time, csv

import matplotlib.pyplot as plt
from matplotlib import animation

import serial

# Tensorflow / Keras
import tensorflow as tf
from tensorflow import keras # for building Neural Networks
print('Tensorflow/Keras: %s' % keras.__version__) # print version

def readData():
    buffer = ""
    while True:
        oneByte = ser.read(1)
        if oneByte == b"\n":
            return buffer
        else:
            buffer += oneByte.decode("ascii")

plot_alpha = 0.25

plt.close('all')

timestr = time.strftime('%d%b%Y_%I%M%p')

file_type = r'.csv'

folder_path = r'C:\Your folder here'

training_model_sub_folder = r'Keras_Model_25Jan2023_0003AM'

plotAlpha = 0.25

#Include the timestamps in the data
include_timestamp_boolean = True

#Lazy workaround to remove trailing carriage return from ESP32 
remove_trailing_cr_bool = True

#Show the serial data in the Python console
show_serial_data_boolean = True

#Option to log the raw data collected from the sensors
log_raw_data_bool = True

#Measure the time it takes to take one reading and feed it to the neural network.
processing_duration_meas = 0

index_curr = 0

#Define the (normalized) bounds for the target on the rebounder, with respect to
#which an impact is considered "good," "marginal," or "bad."
bound_percent = 0.05
x_bounds_good = [0.5 - bound_percent, 0.5 + bound_percent]
x_bounds_marginal = [0.5 - bound_percent*2, 0.5 + bound_percent*2]
y_bounds_good = [0.5 - bound_percent, 0.5 + bound_percent]
y_bounds_marginal = [0.5 - bound_percent*2, 0.5 + bound_percent*2]

#Number of data points used to define an impact.
#Must match the number used for the neural network script.
N = 12

#Marker used for helping to zero the accelerometer readings
first_values_recorded_bool = False
initial_val_1 = 0 #Placeholder
initial_val_2 = 0
initial_val_3 = 0
initial_val_4 = 0

raw_values_received_bool = True
plot_data_when_finished_bool = True

#If an RNN was used, you may need to reshape the data before feeding it into the
#neural net.
RNN_bool = True

successful_exit_bool = False

#Implementation of algorithm from https://stackoverflow.com/a/22640362/6029703
#lag = the lag of the moving window (use the last N observations to smooth the data)
#threshold = the z-score at which the algorithm signals (e.g. threshold of 3.5 means signal will trigger if datapoint is 3.5 std devs away from the moving mean)
#influence = the influence (between 0 and 1) of new signals on the mean and standard deviation (influence of 0 ignores signals when calculating the new threshold)
def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))


#Among the impact period "islands," grab the maximum radius to identify the
#peak of the impact.
#https://stackoverflow.com/questions/50151417/numpy-find-indices-of-groups-with-same-value
def islandinfo(y, trigger_val, stopind_inclusive=True):
    # Setup "sentients" on either sides to make sure we have setup
    # "ramps" to catch the start and stop for the edge islands
    # (left-most and right-most islands) respectively
    y_ext = np.r_[False,y==trigger_val, False]

    # Get indices of shifts, which represent the start and stop indices
    idx = np.flatnonzero(y_ext[:-1] != y_ext[1:])

    # Lengths of islands if needed
    lens = idx[1::2] - idx[:-1:2]

    # Using a stepsize of 2 would get us start and stop indices for each island
    return list(zip(idx[:-1:2], idx[1::2]-int(stopind_inclusive))), lens

try:
    #The pre-trained model has the filetype ".pb" along with some folders of
    #other assets for the model, if applicable.
    #https://www.tensorflow.org/guide/saved_model
    #The saved model can be loaded using tf.saved_model.load(file_path)
    model_folder = folder_path + '\\' + training_model_sub_folder
    
    #The loss function that was used for training only needs to be included if
    #you want to compile the model, not if you just want to make predictions/
    #inferences, which is what we want here. So, set compile = False
    model = tf.keras.models.load_model(model_folder + '\\',compile=False)
    
    print('Pre-trained neural network model loaded.')
    
    print()
    print('Initializing ESP32...')
    
    file_prefix = r'Inference_Results'
    file_type = '.csv'
    file_name = file_prefix + '_' + timestr
    file_path = folder_path + '\\' + file_name + file_type
    
    addr = "COM7"
    baud = 9600
    
    ser = serial.Serial(
        port = addr,\
        baudrate = baud,\
        parity=serial.PARITY_NONE,\
        stopbits=serial.STOPBITS_ONE,\
        bytesize=serial.EIGHTBITS,\
        timeout=0)
    
    print("Connected to: " + ser.portstr)
    
    wait_sec = 1
    print('Wait for ' + str(wait_sec) + ' seconds...')
    temp = time.time()
    while (time.time() - temp < wait_sec):
        0 #Do nothing
    
    #Test that the data can be read
    print('Testing that one line can be read from the ESP32...')
    print('If this line does not proceed, unplug and re-plug the ESP32 USB cable, then run this script again.')
    print('Also check that there arent any debugging print statements enabled in the ESP32 code.')
    print('If that doesnt work, re-flash the code onto the ESP32.')
    line = readData()
    print('Confirmed!')
    
    seq = []
    count = 1
    
    start_time = time.time()
    
    impact_data_file_name = 'Real_Time_Impact_Data' +  '_' + timestr
    impact_data_file_path = folder_path + '\\' + impact_data_file_name + file_type
    
    #Format to write to multiple CSV files in a loop:
    #https://stackoverflow.com/questions/37015439/how-to-write-results-in-a-for-loop-to-multiple-csv-files
    with open(file_path, 'w', newline='') as csvfile, open(impact_data_file_path, 'w', newline='') as csvfile2:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter2 = csv.writer(csvfile2, delimiter=',')
        
        print('Writing header to results file...')
        #Write the header
        spamwriter.writerow(['Time[sec]','X[%]','Y[%]'])
        
        if log_raw_data_bool:
            spamwriter2.writerow(['Time[sec]','Sensor 1','Sensor 2','Sensor 3','Sensor 4'])
        
        print('Done.')
        print('Now waiting for impacts...')
        
        while True:
            
            time_curr = time.time() - start_time
            
            #Take readings and discard them until the final sensor is read, then proceed to read the next line.
            #This should work out such that the next sensor reading is reading #1.
            proceed_bool = False
            while proceed_bool == False:
                line = readData()
                #print(line)
                if line[0] == '4':
                    proceed_bool = True
            
            sensor_vals_curr_1 = []
            sensor_vals_curr_2 = []
            sensor_vals_curr_3 = []
            sensor_vals_curr_4 = []
            
            if first_values_recorded_bool == False:
                line = readData()
                initial_val_1 = int(line[2:])
                line = readData()
                initial_val_2 = int(line[2:])
                line = readData()
                initial_val_3 = int(line[2:])
                line = readData()
                initial_val_4 = int(line[2:])
                
                first_values_recorded_bool = True
            
            line = readData()
            sensor_vals_curr_1.append(int(line[2:]) - initial_val_1)
            line = readData()
            sensor_vals_curr_2.append(int(line[2:]) - initial_val_2)
            line = readData()
            sensor_vals_curr_3.append(int(line[2:]) - initial_val_3)
            line = readData()
            sensor_vals_curr_4.append(int(line[2:]) - initial_val_4)
            
            #This value must match that used in Rebounder_Post_Processor_V3_Clean.py
            acc_mag_peak_threshold = 5000
            
            threshold_bool = (sensor_vals_curr_1[-1] >= acc_mag_peak_threshold) or (sensor_vals_curr_2[-1] >= acc_mag_peak_threshold) or (sensor_vals_curr_3[-1] >= acc_mag_peak_threshold) or (sensor_vals_curr_4[-1] >= acc_mag_peak_threshold)
            if threshold_bool:
                #Grab the remaining N-1 value and feed them into the neural network to make an impact position guess.
                for i in range(0,N-1):
                    line = readData()
                    sensor_vals_curr_1.append(int(line[2:]) - initial_val_1)
                    line = readData()
                    sensor_vals_curr_2.append(int(line[2:]) - initial_val_2)
                    line = readData()
                    sensor_vals_curr_3.append(int(line[2:]) - initial_val_3)
                    line = readData()
                    sensor_vals_curr_4.append(int(line[2:]) - initial_val_4)
                if log_raw_data_bool:
                    curr_row = [time_curr] + sensor_vals_curr_1 + sensor_vals_curr_2 + sensor_vals_curr_3 + sensor_vals_curr_4
                    spamwriter2.writerow(curr_row)
            
                data_for_nn = np.array([sensor_vals_curr_1 + sensor_vals_curr_2 + sensor_vals_curr_3 + sensor_vals_curr_4])
                
                #Put the list of data inside another list, because that's
                #how I structured it during training.
                print()
                infer_landing_bool = True
                display_inference_duration_boolean = False
                
                if RNN_bool:
                    data_for_nn = np.reshape(data_for_nn, (data_for_nn.shape[0], 1, data_for_nn.shape[1]))

                if display_inference_duration_boolean:
                    if infer_landing_bool: neural_net_output = model.predict(data_for_nn)
                else:
                    if infer_landing_bool: neural_net_output = model.predict(np.array(data_for_nn), verbose=0)
                
                x_out = neural_net_output[:,0][0]
                y_out = neural_net_output[:,1][0]
                
                #If x or y is outside the marginal bounds, it's a bad impact
                if x_out < x_bounds_marginal[0] or x_out > x_bounds_marginal[1] or y_out < y_bounds_marginal[0] or y_out > y_bounds_marginal[1]:
                    print('Bad impact detected')
                    ser.write(bytes("0", 'utf-8'))
                else:
                    #If both x and y are inside the good bounds, it's a good impact
                    if x_out > x_bounds_good[0] and x_out < x_bounds_good[1] and y_out > y_bounds_good[0] and y_out < y_bounds_good[1]:
                        print('Good impact detected')
                        ser.write(bytes("2", 'utf-8'))
                    #Otherwise, it must be marginal.
                    else:
                        print('Marginal impact detected')
                        ser.write(bytes("1", 'utf-8'))
                
                print("x_out, y_out: " + str(x_out) + ", " + str(y_out))
                spamwriter.writerow([curr_row[0],x_out,y_out])
                    
                    

except KeyboardInterrupt:
    
    print('Cntl+C press detected - ending script.')
    
    print()
    print('Impact result data saved to:')
    print(file_name)
    
    if log_raw_data_bool:
        print('Raw data saved to:')
        print(impact_data_file_name)

    ser.close()
    
    end_time = time_curr
    successful_exit_bool = True
    
except Exception as e:
    
    print('Other error occured - ending script.')
    print(e)
    
    print()
    print('Log data saved to:')
    print(file_name)

    ser.close()
    
    end_time = time_curr

if successful_exit_bool:
    if plot_data_when_finished_bool:
    
        print('Reading data from file: ' + impact_data_file_name + file_type)
        
        time_data = []
        accel_data_1 = []
        accel_data_2 = []
        accel_data_3 = []
        accel_data_4 = []
        
        raw_data = []
        
        with open(impact_data_file_path) as csvfile:
                reader = csv.reader(csvfile,delimiter=',')
                for row in reader:
                    raw_data.append(row)
        
        raw_data_2 = []
        start_row_index = 0
        
        for i in range(start_row_index,len(raw_data)-1):
            row_curr = []
            #temp = raw_data[i][0].split(',')
            
            temp = raw_data[i]
            
            remove_trailing_newline_bool = True
            if remove_trailing_newline_bool:
                temp[-1] = temp[-1][:-1]
            for j in range(0,len(temp)):
                row_curr.append(float(temp[j]))
            #row_curr = np.array(row_curr)
            
            time_data.append(row_curr[0])
            accel_data_1.append(row_curr[1:7])
            accel_data_2.append(row_curr[7:13])
            accel_data_3.append(row_curr[13:19])
            accel_data_4.append(row_curr[19:25])
            
            raw_data_2.append(row_curr)
        
        accel_data_1 = np.array(accel_data_1)
        accel_data_2 = np.array(accel_data_2)
        accel_data_3 = np.array(accel_data_3)
        accel_data_4 = np.array(accel_data_4)
        
        #Calculate the acceleration magnitudes over time
        accel_mag_1_raw = np.sqrt(np.power(accel_data_1[:,3],2) + np.power(accel_data_1[:,4],2) + np.power(accel_data_1[:,5],2))
        accel_mag_2_raw = np.sqrt(np.power(accel_data_2[:,3],2) + np.power(accel_data_2[:,4],2) + np.power(accel_data_2[:,5],2))
        accel_mag_3_raw = np.sqrt(np.power(accel_data_3[:,3],2) + np.power(accel_data_3[:,4],2) + np.power(accel_data_3[:,5],2))
        accel_mag_4_raw = np.sqrt(np.power(accel_data_4[:,3],2) + np.power(accel_data_4[:,4],2) + np.power(accel_data_4[:,5],2))
        
        fig,ax = plt.subplots()
        plt.title(impact_data_file_name + '\n' + 'Acceleration Values')
        plt.xlabel('Time [sec]')
        plt.ylabel('Accel Value')
        plt.grid(True,alpha=plot_alpha)
        plt.plot(time_data,accel_mag_1_raw,'-s',label = 'Sensor 1 Raw Data')
        plt.plot(time_data,accel_mag_2_raw,'-s',label = 'Sensor 2 Raw Data')
        plt.plot(time_data,accel_mag_3_raw,'-s',label = 'Sensor 3 Raw Data')
        plt.plot(time_data,accel_mag_4_raw,'-s',label = 'Sensor 4 Raw Data')
        plt.legend()
        plt.savefig(folder_path + '\\' + 'AccelMag_DuringInference' + '.png', dpi=200)
        
        
        
        
        
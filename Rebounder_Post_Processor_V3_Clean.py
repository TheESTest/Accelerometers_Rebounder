# -*- coding: utf-8 -*-
r"""
Elation Sports Technologies LLC
www.elationsportstechnologies.com

This script post-processes the data collected by the rebounder accelerometers
and the webcam ball tracking.


"""

import csv,time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import matplotlib.cm as cm
import scipy.interpolate as scipyint
import pylab
import pickle

def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

plot_alpha = 0.25

plt.close('all')

#https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy
#x is the data you want to take the rolling average for.
#w is the size of the window to take the average
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

rolling_avg_window = 3

timestamp_included_bool = True #Assume the timestamp is the first column
remove_trailing_newline_bool = False
plot_individual_files_boolean = True

timestr = time.strftime("%d%b%Y_%H%M%p")

folder_path = r'C:\Your folder file path here'

animation_folder = folder_path

file_type = '.csv'

#Multiple log files can be processed if you wish. To do so, append their
#names to this list and ensure that all the files are located in the same
#folder on your computer.
file_names_to_process = []
file_names_to_process.append(r'Log_23Jan2023_0804PM')
file_names_to_process.append(r'Log_24Jan2023_1014PM')

header = 'd1_amag,d2_amag,d3_amag,d4_amag'
header_row = ['d1_amag','d2_amag','d3_amag','d4_amag']

#Grab N number of points from each of the 4 x accelerometers to adequately define the shape of an impact
N = 12
header_row = []
for i in range(1,5):
    for j in range(0,N):
        header_row.append('d' + str(i) + '_amag_' + str(j))
header_row.append('c_x')
header_row.append('c_y')
header_row.append('c_r')

data_type_string = 'acc_magnitudes'

file_labels = []

all_data = []

output_suffix = 'parsed'

for k in range(0,len(file_names_to_process)):
    
    plt.close('all')
    
    file_name_to_process = file_names_to_process[k]

    file_path_to_process = folder_path + '\\' + file_name_to_process + file_type
    file_path_output= folder_path + '\\' + file_name_to_process + '_' + output_suffix + file_type
    
    print('Reading data from file: ' + file_name_to_process + file_type)
    
    accel_data_1 = []
    accel_data_2 = []
    accel_data_3 = []
    accel_data_4 = []
    webcam_data = []
    
    raw_data = []
    
    with open(file_path_to_process) as csvfile:
            reader = csv.reader(csvfile,delimiter=',')
            for row in reader:
                raw_data.append(row)
    
    raw_data_2 = []
    start_row_index = 2
    
    for i in range(start_row_index,len(raw_data)-1):
        row_curr = []
        temp = raw_data[i][0].split(',')
        if remove_trailing_newline_bool:
            temp[-1] = temp[-1][:-1]
        for j in range(0,len(temp)):
            row_curr.append(float(temp[j]))
        #row_curr = np.array(row_curr)
        
        if row_curr[1] == 1: accel_data_1.append(row_curr)
        if row_curr[1] == 2: accel_data_2.append(row_curr)
        if row_curr[1] == 3: accel_data_3.append(row_curr)
        if row_curr[1] == 4: accel_data_4.append(row_curr)
        if row_curr[1] == 5: webcam_data.append(row_curr)
        
        raw_data_2.append(row_curr)
    
    accel_data_1 = np.array(accel_data_1)
    accel_data_2 = np.array(accel_data_2)
    accel_data_3 = np.array(accel_data_3)
    accel_data_4 = np.array(accel_data_4)
    webcam_data = np.array(webcam_data)
    
    target_fps = 24
    
    #Determine the start and end times used for interpolating all the data sets
    first_time_sec = max([min(accel_data_1[:,0]), min(accel_data_2[:,0]), min(accel_data_3[:,0]), min(accel_data_4[:,0]), min(webcam_data[:,0])])
    end_time_sec = min([max(accel_data_1[:,0]), max(accel_data_2[:,0]), max(accel_data_3[:,0]), max(accel_data_4[:,0]), max(webcam_data[:,0])])
    
    times_new = np.linspace(first_time_sec,end_time_sec,int(target_fps*10*(end_time_sec-first_time_sec)))
    
    #Placeholders
    accel_data_1_new = [0,0,0]
    accel_data_2_new = [0,0,0]
    accel_data_3_new = [0,0,0]
    accel_data_4_new = [0,0,0]
    webcam_data_new = [0,0,0,0,0]
    
    accel_data_1_new[0] = times_new
    accel_data_2_new[0] = times_new
    accel_data_3_new[0] = times_new
    accel_data_4_new[0] = times_new
    webcam_data_new[0] = times_new
    
    indices_list = [2]
    for i in indices_list:
        #Zero the accelerometer data
        accel_data_1[:,i] = accel_data_1[:,i] - accel_data_1[0,i]
        accel_data_2[:,i] = accel_data_2[:,i] - accel_data_2[0,i]
        accel_data_3[:,i] = accel_data_3[:,i] - accel_data_3[0,i]
        accel_data_4[:,i] = accel_data_4[:,i] - accel_data_4[0,i]
        
        #Perform linear interpolation
        f1 = scipyint.interp1d(accel_data_1[:,0], accel_data_1[:,i], kind='linear')
        f2 = scipyint.interp1d(accel_data_2[:,0], accel_data_2[:,i], kind='linear')
        f3 = scipyint.interp1d(accel_data_3[:,0], accel_data_3[:,i], kind='linear')
        f4 = scipyint.interp1d(accel_data_4[:,0], accel_data_4[:,i], kind='linear')
        accel_data_1_new[i] = f1(times_new)
        accel_data_2_new[i] = f2(times_new)
        accel_data_3_new[i] = f3(times_new)
        accel_data_4_new[i] = f4(times_new)
        
    
    #Circle tracking data
    for i in [2,3,4]:
        #Linear fit
        f5 = scipyint.interp1d(webcam_data[:,0], webcam_data[:,i], kind='linear')
        webcam_data_new[i] = f5(times_new)
        
    webcam_data = np.array(webcam_data)
        
    circle_data_t = webcam_data_new[0]
    circle_data_x = webcam_data_new[2]
    circle_data_y = webcam_data_new[3]
    circle_data_r = webcam_data_new[4]
    
    #Calculate magnitudes of accelerations
    accel_mag_1_raw = np.array(accel_data_1[:,2])
    accel_mag_2_raw = np.array(accel_data_2[:,2])
    accel_mag_3_raw = np.array(accel_data_3[:,2])
    accel_mag_4_raw = np.array(accel_data_4[:,2])
    
    accel_mag_1 = accel_mag_1_raw
    accel_mag_2 = accel_mag_2_raw
    accel_mag_3 = accel_mag_3_raw
    accel_mag_4 = accel_mag_4_raw
    
    #Truncate the data as needed so that all 4 x accelerometer data streams are
    #the same length.
    min_length = np.min([len(accel_mag_1_raw),len(accel_mag_2_raw),len(accel_mag_3_raw),len(accel_mag_4_raw)])
    accel_data_1 = accel_data_1[:min_length]
    accel_data_2 = accel_data_2[:min_length]
    accel_data_3 = accel_data_3[:min_length]
    accel_data_4 = accel_data_4[:min_length]
    accel_mag_1_raw = accel_mag_1_raw[:min_length]
    accel_mag_2_raw = accel_mag_2_raw[:min_length]
    accel_mag_3_raw = accel_mag_3_raw[:min_length]
    accel_mag_4_raw = accel_mag_4_raw[:min_length]
    
    #Identify the starts of the impacts from the accelerometer magnitude data.    
    impact_start_and_end_indices = []
    impact_start_and_end_times = []
    impact_start_times = []
    impacts_data = []
    impact_data_curr = [[],[],[],[]]
    
    acc_mag_peak_threshold = 5000
    found_first_instance_bool = False
    i = 0
    print('Grabbing impact data...')
    while i < min_length:
        threshold_bool = (accel_mag_1_raw[i] >= acc_mag_peak_threshold) or (accel_mag_2_raw[i] >= acc_mag_peak_threshold) or (accel_mag_3_raw[i] >= acc_mag_peak_threshold) or (accel_mag_4_raw[i] >= acc_mag_peak_threshold)
        
        if (threshold_bool) and (i + N < min_length):
            
            impact_start_and_end_indices.append([i,i+N])
            curr_start_time = np.min([accel_data_1[:,0][i],accel_data_2[:,0][i],accel_data_3[:,0][i],accel_data_4[:,0][i]])
            curr_end_time = np.min([accel_data_1[:,0][i+N],accel_data_2[:,0][i+N],accel_data_3[:,0][i+N],accel_data_4[:,0][i+N]])
            impact_start_and_end_times.append([curr_start_time,curr_end_time])
            
            #Grab the next N number of data points
            impact_data_curr[0] = accel_mag_1_raw[i:i+N]
            impact_data_curr[1] = accel_mag_2_raw[i:i+N]
            impact_data_curr[2] = accel_mag_3_raw[i:i+N]
            impact_data_curr[3] = accel_mag_4_raw[i:i+N]
            impacts_data.append(impact_data_curr)
            impact_data_curr = [[],[],[],[]]
            
            i+=N
                    
        else:
            i+=1
        
    impacts_data = np.array(impacts_data)


    #Identify the maximum radius peaks from the webcam data using smoothed z-scores algorithm.
    #Implementation of algorithm from https://stackoverflow.com/a/22640362/6029703
    #lag = the lag of the moving window (use the last N observations to smooth the data)
    #threshold = the z-score at which the algorithm signals (e.g. threshold of 3.5 means signal will trigger if datapoint is 3.5 std devs away from the moving mean)
    #influence = the influence (between 0 and 1) of new signals on the mean and standard deviation (influence of 0 ignores signals when calculating the new threshold)
    
    lag = 10
    threshold = 3
    influence = 0.01
    
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
    

    y = webcam_data_new[4]
    
    # Run algo with settings from above
    result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)
    
    
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
    
    impact_period_indices = np.where(result['signals'] > 0)[0]
    islands_endcap_indices = islandinfo(result['signals'] > 0, 1)
    
    circle_peak_indices = []
    circle_peak_times = []
    for i in range(0,len(islands_endcap_indices[0])):
        index_pair_curr = islands_endcap_indices[0][i]
        
        #If an island only has 1 x element, ignore it.
        if index_pair_curr[0] != index_pair_curr[1]:
            data_curr = webcam_data_new[4][index_pair_curr[0]:index_pair_curr[1]]
            peak_index_curr = np.argmax(data_curr) + index_pair_curr[0]
            circle_peak_indices.append(peak_index_curr)
    
    #Now the circle-tracking peaks are identified. For all the impact start times
    #as determined by the accelerometer data, find the closest-matching circle-tracking
    #impact info (x,y,rad).
    #I use the accelerometer impact start times (not the end times) as the reference
    #because I've found that the accelerometer impact data tends to lag the webcam data.
    #https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    def find_nearest_and_index(array, value):
        array = np.array(array)
        idx = np.argmin(np.abs(array - value))
        return array[idx], idx
    
    corresponding_circle_impact_data = []
    for i in range(0,len(impact_start_and_end_times)):
        curr_start_time = impact_start_and_end_times[i][0]
        
        curr_circle_peak_time, index_temp = find_nearest_and_index(webcam_data_new[0][circle_peak_indices], curr_start_time)
                
        curr_circle_x = np.interp(curr_circle_peak_time,webcam_data_new[0],webcam_data_new[2])
        curr_circle_y = np.interp(curr_circle_peak_time,webcam_data_new[0],webcam_data_new[3])
        curr_circle_r = np.interp(curr_circle_peak_time,webcam_data_new[0],webcam_data_new[4])
        
        curr_circle_data = [curr_circle_x,curr_circle_y,curr_circle_r]
        corresponding_circle_impact_data.append(curr_circle_data)
    

    #With all the data for the impacts identified:
    #1) Concatenate the accelerometer data (i.e. all the sensor #1 datapoints, followed by sensor #2, etc.)
    #2) Then, tack on the ball landing position data (i.e. x,y,radius)    
    impacts_dataset = []
    #Impact number
    for i in range(0,len(impacts_data)):
        impact_data_curr = []
        #Sensor number
        for j in range(0,4):
            #Datapoint number
            for k in range(0,len(impacts_data[0][0])):
                impact_data_curr.append(impacts_data[i][j][k])
        curr_circle_data = corresponding_circle_impact_data[i]
        impact_data_curr.append(curr_circle_data[0]) #x
        impact_data_curr.append(curr_circle_data[1]) #y
        impact_data_curr.append(curr_circle_data[2]) #radius
        impacts_dataset.append(impact_data_curr)
    
            
    all_data.append(impacts_dataset)
    
    #Save the current file's training dataset to an CSV file
    with open(folder_path + '\\' + file_name_to_process + '_' + 'Training_Data' + '.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(header_row)
        for elem in impacts_dataset:
            spamwriter.writerow(elem)
    
    
    impacts_dataset = np.array(impacts_dataset)
    impacts_x = impacts_dataset[:,-3]
    impacts_y = impacts_dataset[:,-2]
    impacts_r = impacts_dataset[:,-1]
    
    #I want to make my plots and calculations at high resolution, but only animate at the target FPS
    times_anim = np.linspace(first_time_sec,end_time_sec,int(target_fps*(end_time_sec-first_time_sec)))
    circle_data_x_anim = np.interp(times_anim,circle_data_t,circle_data_x)
    circle_data_y_anim = np.interp(times_anim,circle_data_t,circle_data_y)
    circle_data_r_anim = np.interp(times_anim,circle_data_t,circle_data_r)
    
    impact_start_and_end_times = np.array(impact_start_and_end_times)
    
    print('Making plots...')
    
    #Plot a representative set of curves describing one of the impacts
    #Note the time steps are approximated, i.e. end time minus start time, divided by the total number of points (e.g. 12)
    rand_ind = np.random.randint(len(impacts_data))
    start_time_curr = impact_start_and_end_times[rand_ind][0]
    end_time_curr = impact_start_and_end_times[rand_ind][1]
    time_array_approx = np.linspace(start_time_curr,end_time_curr,N)
    fig,ax = plt.subplots()
    plt.title(file_name_to_process + '\n' + 'Representative Impact Curves, Impact # ' + str(rand_ind) + ' at time t = ' + str(round(start_time_curr)))
    plt.xlabel('Time [sec]')
    plt.ylabel('Sensors Output')
    plt.grid(True,alpha=plot_alpha)
    plt.plot()
    for i in range(0,len(impacts_data[rand_ind])):
        plt.plot(time_array_approx,impacts_data[rand_ind][i],'-o',label='Sensor ' + str(i))
    plt.legend()
    plt.savefig(folder_path + '\\' + file_name_to_process + '_' + 'SingleImpactSpotCheck' + '.png', dpi=200)
    pickle.dump(fig, open(folder_path + '\\' + file_name_to_process + '_' + 'SingleImpactSpotCheck' + '.pkl', 'wb'))  

    fig,ax = plt.subplots()
    plt.title(file_name_to_process + '\n' + 'Ball Radius Over Time')
    plt.xlabel('Time [sec]')
    plt.ylabel('Radius [px]')
    plt.grid(True,alpha=plot_alpha)
    plt.plot(webcam_data_new[0],webcam_data_new[4],'-o',label='Interpolated Data')
    plt.plot(webcam_data[:,0], webcam_data[:,4],'sk',label='Raw Data')
    plt.plot(webcam_data_new[0][circle_peak_indices],webcam_data_new[4][circle_peak_indices],'rs',label='Peaks')
    plt.legend()
    plt.savefig(folder_path + '\\' + file_name_to_process + '_' + 'Radius' + '.png', dpi=200)
    pickle.dump(fig, open(folder_path + '\\' + file_name_to_process + '_' + 'Radius' + '.pkl', 'wb'))  
    
    fig,ax = plt.subplots()
    plt.title(file_name_to_process + '\n' + 'Ball X,Y Landing Positions Over Time')
    plt.xlabel('Time [sec]')
    plt.ylabel('Position [px]')
    plt.grid(True,alpha=plot_alpha)
    plt.plot(webcam_data[:,0], webcam_data[:,2],'-',color='C2',label='X')
    plt.plot(webcam_data[:,0], webcam_data[:,3],':',color='C3',label='Y')
    plt.plot(impact_start_and_end_times[:,0], impacts_x,'sk',label='X Impacts')
    plt.plot(impact_start_and_end_times[:,0], impacts_y,'^k',label='Y Impacts')
    plt.legend()
    plt.savefig(folder_path + '\\' + file_name_to_process + '_' + 'XY_Impacts' + '.png', dpi=200)
    pickle.dump(fig, open(folder_path + '\\' + file_name_to_process + '_' + 'XY_Impacts' + '.pkl', 'wb'))  
    
    
    fig,ax = plt.subplots()
    plt.title(file_name_to_process + '\n' + 'Camera Ball Tracking Over Time')
    plt.xlabel('X [px]')
    plt.ylabel('Y [px]')
    ax.invert_yaxis()
    plt.scatter(webcam_data_new[2],webcam_data_new[3],c=np.arange(0,len(webcam_data_new[0])))
    plt.grid(True,alpha=plot_alpha)
    plt.savefig(folder_path + '\\' + file_name_to_process + '_' + 'XY_Camera' + '.png', dpi=200)
    pickle.dump(fig, open(folder_path + '\\' + file_name_to_process + '_' + 'XY_Camera' + '.pkl', 'wb'))  
    
    
    fig,ax = plt.subplots()
    plt.title(file_name_to_process + '\n' + 'Acceleration Values')
    plt.xlabel('Time [sec]')
    plt.ylabel('Accel Value')
    plt.grid(True,alpha=plot_alpha)
    plt.plot(accel_data_1[:,0],accel_mag_1_raw,'s-',label = 'Sensor 1')
    plt.plot(accel_data_2[:,0],accel_mag_2_raw,'s-',label = 'Sensor 2')
    plt.plot(accel_data_3[:,0],accel_mag_3_raw,'s-',label = 'Sensor 3')
    plt.plot(accel_data_4[:,0],accel_mag_4_raw,'s-',label = 'Sensor 4')
    ymin_curr = min(ax.get_ylim())
    ymax_curr = max(ax.get_ylim())
    start_time_curr = impact_start_and_end_times[0][0]
    plt.plot([start_time_curr,start_time_curr],[ymin_curr,ymax_curr],'--k',label='Data Spike Detected')
    for i in range(1,len(impact_start_and_end_times)):
        start_time_curr = impact_start_and_end_times[i][0]
        plt.plot([start_time_curr,start_time_curr],[ymin_curr,ymax_curr],'--k')
    plt.legend()
    plt.savefig(folder_path + '\\' + file_name_to_process + '_' + 'AccelMag' + '.png', dpi=200)
    pickle.dump(fig, open(folder_path + '\\' + file_name_to_process + '_' + 'AccelMag' + '.pkl', 'wb'))  
    
    

    
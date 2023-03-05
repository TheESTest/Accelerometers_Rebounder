# -*- coding: utf-8 -*-
r"""
Elation Sports Technologies LLC
https://www.elationsportstechnologies.com/

Combine CSV files provided in a list.


"""

import csv
import numpy as np

folder_path = r'C:\Your folder path here'

file_type = r'.csv'

print('Reading files...')

file_names = []
file_names.append(r'Log_19Jan2023_0818PM_Training_Data')
file_names.append(r'Log_19Jan2023_0937PM_Training_Data')

combined_list = []

include_header_boolean = True

for i in range(0,len(file_names)):
    
    file_name = file_names[i]

    print('Reading file: ' + file_name + file_type)
    
    raw_data = []
    
    file_path = folder_path + '\\' + file_name + file_type
    
    with open(file_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            raw_data.append(row)
    
    header = raw_data[0]
    
    start_row = 1
    end_row = len(raw_data)-1
    for i in range(start_row,end_row):
        curr_row = raw_data[i]
        curr_list = []
        for elem in curr_row:
            curr_list.append(float(elem))
        combined_list.append(curr_list)
    


combined_list = np.array(combined_list)

print()
print('Combined list:')
print(combined_list)

#Save the current file's training dataset to an CSV file
with open(folder_path + '\\' + 'Combined_Training_Data.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    if include_header_boolean: spamwriter.writerow(header)
    for row in combined_list:
        spamwriter.writerow(row)





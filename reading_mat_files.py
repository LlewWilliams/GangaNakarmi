'''
Ganga Nakarmi
West Virginia University
Coding assistance from Llew Williams

This code was used in collecting data for this research paper:
   "A crowdsourced approach to documenting users’ preferences for landscape attributes in the proposed Appalachian Geopark Region in West Virginia"

This was used to read from matlab files produced by CRF processing
'''

from scipy.io import loadmat
mat_file_name = "*.mat"
import os
import csv
import numpy as np


location_of_output_file = 'OUTPUT_FILE_LOCATION' #file location can be added here or input from prompt later

#output file name
output_file_name = 'OUTPUT_FILE_NAME.csv' #this is default file - it could be edited 

#input file name
input_file_name = r'INPUT_FILE_NAME.mat'

#location of input file
location_of_input_file_name = r'INPUT_FILE_LOCATION'

# to create a new file
first_time = True


# working array length adjustment
# NOTE: We were working with 7 categories
t = [0,1,2,3,4,5,7,7,7,8] 
p = [0,1,2,3,4,5,7,7]
if len(t) != len(p):
    arrays = [t,p]
    max_length = 0
    for array in arrays:
        max_length = max(max_length, len(array))
    for array in arrays:
        if len(array) < max_length:
            len_small_array = len(array)
            diff = max_length - len_small_array 
            print("detected different list length. Removed " + str(diff) + " class values")
    for array in arrays:
        if len(array) == max_length:
            array = array[0:len_small_array]
    if len(t) > len(p):
        t=t[0:len_small_array]
    else:
        p=p[0:len_small_array]

print("t",t)
print("p", p)
       

def write_csv_file(name,classes):
    output_file_location_and_name = os.path.join(location_of_output_file,output_file_name)
    if not first_time:
        out_file = open(output_file_location_and_name, 'a',encoding="utf-8",newline='')
    else:
        out_file = open(output_file_location_and_name, 'w',encoding="utf-8",newline='')
    if classes.any():
        writer = csv.writer(out_file)
        counter = 0 
        list_to_write = []
        zero = 0
        one = 0
        two = 0
        three = 0
        four = 0
        five = 0 
        six = 0 
        if first_time==True:
            writer.writerow(['name','zero','one','two','three','four','five','six','total'])
        for i in range(len(classes)):            
            for j in range(len(classes[i])):
                counter+=1
                class_value = classes[i][j]
                if class_value == 0:
                    zero += 1
                elif class_value == 1:
                    one += 1
                elif class_value == 2:
                    two += 1
                elif class_value == 3:
                    three += 1
                elif class_value == 4:
                    four += 1
                elif class_value == 5:
                    five += 1
                elif class_value == 6:
                    six += 1
        list_to_write.append(name)
        list_to_write.append(str(zero))
        list_to_write.append(str(one))
        list_to_write.append(str(two))
        list_to_write.append(str(three))
        list_to_write.append(str(four))
        list_to_write.append(str(five))
        list_to_write.append(str(six))
        list_to_write.append(str(counter))
        writer.writerow(list_to_write)
  
    out_file.close()
file_to_load = os.path.join(location_of_input_file_name, input_file_name)

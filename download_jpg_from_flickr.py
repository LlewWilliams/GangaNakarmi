'''
Ganga Nakarmi
West Virginia University
Coding assistance from Llew Williams

This code was used in collecting data for this research paper:
   "A crowdsourced approach to documenting users’ preferences for landscape attributes in the proposed Appalachian Geopark Region in West Virginia"

This was used to download a set of jpg images from Flickr. 
The desired image names and urls were placed in a csv file for processing beforehand. 
'''
import requests
import csv
import os
# need path to project folder
project_folder = os.path.abspath(os.path.dirname(__file__))

#csv folder
csv_folder = 'DESIRED_FOLDER'
csv_folder_path = os.path.join(project_folder, csv_folder)
csv_file_name = 'DESIRED_FILE_NAME.csv'
csv_folder_path_and_file = os.path.join(csv_folder_path, csv_file_name)


with open(csv_folder_path_and_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count != 0:
            url = row[4]
            line_count += 1
            trimmed_file_name = os.path.basename(row[4])
            image_download_folder = os.path.join(project_folder,'RELATIVE_PATH_TO_OUTPUT_FOLDER')
            trimmed_file_name_and_path = os.path.join(image_download_folder,trimmed_file_name)
            r = requests.get(url, allow_redirects=True)           
        else:
            line_count += 1


    print('Processed ' + str(line_count) + ' lines.')



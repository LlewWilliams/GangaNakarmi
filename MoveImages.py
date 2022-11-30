
import os
import csv
# read names from csv file
input_file_name = 'INPUT_FILE.csv'
location_of_input_file = r'INPUT_FILE_LOCATION'
current_working_directory = os. getcwd()
input_file_location_and_name = os.path.join(current_working_directory,'FOLDERNAME/'+input_file_name)

#images
image_path = r'PATH_TO_IMAGES'


with open(input_file_location_and_name,'r') as file:
        reader = csv.reader(file)
        #columns = next(reader) 
        #columns = columns[1:]
       
        for data in reader:
            #data = data[8]
            #image_data = data[0]+  '.jpg'
            image_data = data[0]
            current_image_path = os.path.join(image_path,image_data )
            moved_file=current_image_path.replace('PATH_TO_REPLACE','TARGET_PATH')
            if os.path.exists(current_image_path):
                os.replace(current_image_path, moved_file)
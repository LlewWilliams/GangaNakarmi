'''
Ganga Nakarmi
West Virginia University
Coding assistance from Llew Williams

This code was used in collecting data for this research paper:
   "A crowdsourced approach to documenting users’ preferences for landscape attributes in the proposed Appalachian Geopark Region in West Virginia"

This was used to load landscape classification metrics to a database for analysis 
'''

import csv
import os



import pyodbc
server = 'SERVER_NAME'
database = 'DATABASE_NAME'
username = 'USER_NAME'
password = 'PWD'   
driver= 'ODBC_DRIVER'
connection_string = 'DRIVER='+driver+';SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password #Trusted_Connection=yes;'

input_file_name = 'INPUT_FILE.csv'

location_of_input_file = r'INPUT_FILE_LOCATION'
input_file_location_and_name = os.path.join(location_of_input_file,input_file_name)
with pyodbc.connect(connection_string) as conn:
    with open(input_file_location_and_name,'r') as file:
        reader = csv.reader(file)
        columns = next(reader) 
        #columns = columns[1:]
        #file 1 - this is csv from semseg_cnn_crf
        #query = 'insert into class_data({0}) values ({1})'
        #file 2
        query = 'INSERT INTO CRF_Contrast_class_data({0}) values ({1})'
        query = query.format(','.join(columns), ','.join('?' * len(columns)))
        cursor = conn.cursor()
        for data in reader:
            #data = data[1:]
            cursor.execute(query, data)
        cursor.commit()
  


'''
Ganga Nakarmi
West Virginia University
Coding assistance from Llew Williams

This code was used in collecting data for this research paper:
   "A crowdsourced approach to documenting users’ preferences for landscape attributes in the proposed Appalachian Geopark Region in West Virginia"

This was a utility that we used to move images around on the file system during processing
'''

import csv
import os
import shutil



import pyodbc

server = 'SERVER_NAME'
database = 'DATABASE_NAME'
username = 'USER_NAME'
password = 'PWD'   
driver= 'ODBC_DRIVER'
connection_string = 'DRIVER='+driver+';SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password #Trusted_Connection=yes;'


location_of_images = 'PATH_TO_IMAGES'
place_images_here = 'PATH_TO_IMAGES_TO_PROCESS'
with pyodbc.connect(connection_string) as conn:

    with conn.cursor() as cursor:
        cursor.execute("SQL_STATEMENT")
        row = cursor.fetchone()
        while row:
            fromfile = location_of_images+'/'+str(row[0])+'.jpg'
            tofile = place_images_here +'/'+str(row[0])+'.jpg'
            print (fromfile)
            shutil.copyfile(fromfile,tofile)
            row = cursor.fetchone()




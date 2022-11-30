"""
Ganga Nakarmi
West Virginia University
Coding assistance from Llew Williams

This code was used in collecting data for this research paper:
   "A crowdsourced approach to documenting users’ preferences for landscape attributes in the proposed Appalachian Geopark Region in West Virginia"

This was used to read available file information from Flickr. 
"""
from typing import overload
from flickrapi import FlickrAPI
from pprint import pprint
import csv
# flickr auth information:
# change these to your flickr api keys and secret
flickrAPIKey = "API_KEY"  # API key
flickrSecret = "FLICKR_SECRET"                  # shared "secret"

object_id = 0 #counter for use in csv output
our_file_name_prefix = 'WV' #an idea not used - a way to add an identifier to our files to help identify individual runs of the downloader
photos = {}

def write_csv_file(photos,selection_criteria, our_file_name_prefix, first_time, object_id):
    out_file = open('OUTPUT_FILE_PATH_AND_NAME.csv', 'a',encoding="utf-8",newline='')

    if photos:
        photoList = photos.get('photo')
        if photoList:
            # create the csv writer
            writer = csv.writer(out_file)
            if first_time==True:
                writer.writerow(['OBJECTID','PhotoTakenDate','SelectionCriteria','OurFileName','FlickrFileName','FlickrTitle','Latitude','Longitude'])
            
            for individualPhoto in photoList: 
                if individualPhoto !=None:

                    if individualPhoto.get('url_k') != None:
                        list_to_write = []
                        object_id = object_id + 1
                        list_to_write.append(object_id)
                        #PhotoTakenDate
                        list_to_write.append(individualPhoto.get('datetaken'))
                        #SelectionC #hmm supposed to be SelectionCriteria
                        list_to_write.append(selection_criteria)
                        #OurFileName
                        list_to_write.append(our_file_name_prefix + str(object_id))
                        #FlickrFileName
                        list_to_write.append(individualPhoto.get('url_k'))
                        #FlickrTitle
                        list_to_write.append(individualPhoto.get('title'))
                        #Latitude
                        list_to_write.append(individualPhoto.get('latitude'))
                        #Longitude
                        list_to_write.append(individualPhoto.get('longitude'))
                        writer.writerow(list_to_write)
                #current_image_n
        # close the file
    out_file.close()
    return object_id

# study area is in three counties in southern WV, USA. Same approach could work anywhere with different coordinates
#   Flickr limits the number of hits to prevent crashing their system so we broke up our area into smaller blocks using the approach below. 

maxWestLong = -81.567129
maxEastLong = -79.960126
maxNorthLat = 38.264809
maxSouthLat = 37.507686

# example from New River Bridge -81.086939,38.066328,-81.074483,38.078127
# on map - 15 across - 9 down

incrementLong = (maxWestLong - maxEastLong )/15 #-0.107133
incrementLat = (maxNorthLat - maxSouthLat)/9 #0.084125

lastLong = round(maxWestLong,6) 
lastLat = round(maxSouthLat,6) 
first_time = True
for i in range(14):
    currentLong = round(lastLong - incrementLong, 6)
    for j in range(8):
        currentLat = round(lastLat + incrementLat, 6)
        bounding_string = str(lastLong)+','+str(lastLat)+','+str(currentLong)+','+str(currentLat)
        selection_criteria = bounding_string.replace(',','_')
        lastLat= currentLat
        # call to get pics with api  goes here
        flickr = FlickrAPI(flickrAPIKey, flickrSecret, format='parsed-json')
        extras='geo,url_k,date_taken'
        #min_date_taken='yyyy-mm-dd'
        max_date_taken='2020-12-31' # our time frame of interest
        foundPictures = flickr.photos.search(bbox=bounding_string,max_taken_date=max_date_taken, has_geo=1, extras=extras)
        photos = foundPictures.get('photos')
        
        pprint(photos)
        if photos:
            object_id = write_csv_file(photos, bounding_string.replace(',','_'), our_file_name_prefix, first_time, object_id)
            first_time = False

    lastLat = round(maxSouthLat,6)
    lastLong = currentLong

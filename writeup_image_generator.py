import csv 
import numpy as np

# Read csv data
lines = []
with open('/opt/carnd_p3/data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)
        

# labels: ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
# example entry: ['IMG/center_2016_12_01_13_30_48_287.jpg', ' IMG/left_2016_12_01_13_30_48_287.jpg', ' IMG/right_2016_12_01_13_30_48_287.jpg', ' 0', ' 0', ' 0', ' 22.14829']

# #  Splitting Data
data_lines = lines[1:]
# code for generating writeup images
# center driving image
# normal image and flipped normal image
# recovery 1,2,3
i = 0
for line in data_lines[4000:4020]:
    center_filename = line[0].split('/')[-1]
    center_image_path =  './data/IMG/' + center_filename
    center_image = plt.imread(center_image_path)
    left_filename = line[1].split('/')[-1]
    left_image_path =  './data/IMG/' + left_filename
    left_image = plt.imread(left_image_path)
    right_filename = line[2].split('/')[-1]
    right_image_path =  './data/IMG/' + right_filename
    right_image = plt.imread(right_image_path)
    
    plt.imsave('./ifw/data{}'.format(i), center_image)
#     plt.imsave('./ifw/dataflipped{}'.format(i), np.fliplr(center_image))
    plt.imsave('./ifw/dataleft{}'.format(i), left_image)
    plt.imsave('./ifw/dataright{}'.format(i), right_image)
    i+= 1



print("done")
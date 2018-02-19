#Python script for croping images labeled by LabelImg tool (https://github.com/tzutalin/labelImg).
#Change box, basewidth and hsize values according to box label which you want crop form original pictures.
#Images and xml files need to be in same directory. For each image, xml file with same name needs to be defined.




import os
import glob
import PIL
import numpy as np
from PIL import Image



import xml.etree.ElementTree as ET

box = '48'
widthList = []
heightList = []

for filename in sorted(glob.glob('*.xml')):
	for teethdata in ET.parse(filename).getroot().findall('object'):
		if(teethdata.find('name').text == box):
			xmin = int(teethdata.find('bndbox').find('xmin').text)
			ymin = int(teethdata.find('bndbox').find('ymin').text)
			xmax = int(teethdata.find('bndbox').find('xmax').text)
			ymax = int(teethdata.find('bndbox').find('ymax').text)

			widthList.append(xmax-xmin)
			heightList.append(ymax-ymin)

width = int(np.mean(widthList))
height = int(np.mean(heightList))
print(width,height)

#It is recommended to calculate average width and height of each label in order to keep satisfying aspect ratio.


if not os.path.exists('label' + box):
	os.makedirs('label' + box)




for filename in sorted(glob.glob('*.xml')):
	for teethdata in ET.parse(filename).getroot().findall('object'):
		if(teethdata.find('name').text == box):
			xmin = int(teethdata.find('bndbox').find('xmin').text)
			ymin = int(teethdata.find('bndbox').find('ymin').text)
			xmax = int(teethdata.find('bndbox').find('xmax').text)
			ymax = int(teethdata.find('bndbox').find('ymax').text)

			image = Image.open(filename.split('.')[0] + '.jpg')
			cropedImage = image.crop((xmin, ymin, xmax, ymax))
			resizedImage = cropedImage.resize((width,height), Image.ANTIALIAS)
			
			resizedImage.save('label' + box + '/' + filename.split('.')[0] + '_' + box + '.jpg', 'JPEG')



            
            
            

			










		    
		    
		    

		   
		
			
	  









  






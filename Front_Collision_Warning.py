#Import dependencies
import numpy as np
import os
from os.path import exists


import time
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile
from PIL import Image
import glob
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from skimage.metrics import structural_similarity as ssim



import tensorflow as tf
import matplotlib.pyplot as plt
import cv2



# In[6]:


# # Model preparation
#Ppath to TFOD object detection folder
PATH_TO_OBJECT_DETECTION = 'D:/Github/tf_object_detection/models/research/object_detection'
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = MODEL_NAME + '/mscoco_label_map.pbtxt'
NUM_CLASSES = 90



file_exists = exists(PATH_TO_CKPT) and exists(PATH_TO_LABELS)
if file_exists == False:
    # ## Download Model
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

#path to object detection folder
sys.path.append(PATH_TO_OBJECT_DETECTION)

from utils import label_map_util
from utils import visualization_utils as vis_util
from collision_warning import *

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Size, in inches, of the output images.
#IMAGE_SIZE = (12, 8)

#get_ipython().run_line_magic('matplotlib', 'inline')
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        frameNumber = 0
        vid = cv2.VideoCapture('vid1.mp4')
        while True:
        #for image in images:
            start_time = time.time() # start time of the loop to find fps
            #img = cv2.imread(image)
            ret, img = vid.read()
            #img = cv2.imread('./images/1.png')
            img = cv2.resize(img, (800,450))
            image_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channels = image_np.shape
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                      feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=3)
            #draw basic information on feame
            displayInformation(image_np,width,height)
            
            poly_scale = 1.7
            polygone = np.array([[[0, height] , [width, height], [round(width/poly_scale), round(height/poly_scale)],[(width-round(width/poly_scale)), round(height/poly_scale)],]], np.int32)
            polygonePoints = Polygon([(0, height),(width, height),(round(width/poly_scale), round(height/poly_scale)),((width-round(width/poly_scale)), round(height/poly_scale))])
            

            #vehicle speed calculations
            if frameNumber ==0:
                oldFrame = image_np
            grayImage = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            grayOldFrame = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)
            ssimScore = ssim(grayImage, grayOldFrame)
            oldFrame = image_np
            vehicleSpeed = getVehicleSpeed(ssimScore)
            #Display vehichle speed on frame
            cv2.putText(image_np,str(vehicleSpeed)+'Kmph', (720,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (102,255,102), 2)
            cv2.polylines(image_np, [polygone], True, (12,236,236), thickness=2)

            for i,b in enumerate(boxes[0]):
                #                 car                    bus                  truck
                if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
                    if scores[0][i] >= 0.5:

                        #calculate distance of object here
                        bbox = boxes[0][i]
                        width_of_bounding_box = bbox[3]*width - bbox[1]*width
                        distance = round( distance_to_camera(width_of_bounding_box))
                        #print('distance from camera in meters = ',distance)

                        ymin, xmin, ymax, xmax = bbox
                        (left, right, top, bottom) = (xmin * width, xmax * width,
                              ymin * height, ymax * height)

                        #start_point = (round(left),round(top))
                        #end_point = (round(right),round(bottom))
                        #color = (0,255,0)
                        #thickness = 5
                        #poly_scale = 1.7

                        mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                        mid_y = (boxes[0][i][0]+boxes[0][i][2])/2

                        apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)

                        if apx_distance <=0.5:

                            #print(round(left), round(right), round(top), round(bottom))
                            start_point = (round(left), round(top))
                            end_point = ( round(right),round(bottom))
                            #shoe bounding boxes

                            cv2.putText(image_np, str(distance)+'m', (round(right),round(bottom)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (102,255,102), 2)

                            midPoint = Point(round(mid_x*width),round(mid_y*height))
                            flag =checkIfpointIsinPolygone(midPoint,polygonePoints)
                            if flag == True:
                                cv2.putText(image_np, 'WARNING!!!', (330,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 3)
                                image_np = cv2.rectangle(image_np, (323,67), (503,107), (102,255,102), 2)
                                cv2.polylines(image_np, [polygone], True, (255,0,0), thickness=2)
                                image_np = cv2.rectangle(image_np, start_point, end_point, (255,0,0), 2)
                                cv2.putText(image_np, str(distance)+'m', (round(right),round(bottom)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                                if vehicleSpeed != 0:
                                    timeForCollision =round((distance/((vehicleSpeed*1000)/3600)),2)
                                    cv2.putText(image_np,str(timeForCollision) + 'sec', (720,44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (102,255,102), 2)
                                cv2.putText(image_np,'True', (720,66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (102,255,102), 2)
                            #cv2.polylines(image_np, [polygone], True, (0,255,0), thickness=2)
            im_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            #cv2.imwrite(image,im_rgb)

            frameNumber =frameNumber+1
            cv2.imshow('window',cv2.resize(im_rgb,(800,450)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        #plt.imshow(image_np)
        print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop

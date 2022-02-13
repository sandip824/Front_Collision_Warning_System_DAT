import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import cv2

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def distance_to_camera(predicted_Width):
    FOCAL_LENGTH = 841
    KNOWN_WIDTH = 1.5
    # compute and return the distance from the maker to the camera
    return (KNOWN_WIDTH * FOCAL_LENGTH) / predicted_Width
def checkIfpointIsinPolygone(point,poly):
    return poly.contains(point)

def getVehicleSpeed(ssimScore):

    if ssimScore >=0.98:
        return 0
    setSpeed = 60
    calculatedSpeed = setSpeed * ssimScore
    speed = setSpeed-calculatedSpeed
    return int(speed)
    
def displayInformation(image_np,width,height):
    #poly_scale = 1.7
    #polygone = np.array([[[0, height] , [width, height], [round(width/poly_scale), round(height/poly_scale)],[(width-round(width/poly_scale)), round(height/poly_scale)],]], np.int32)
    #polygonePoints = Polygon([(0, height),(width, height),(round(width/poly_scale), round(height/poly_scale)),((width-round(width/poly_scale)), round(height/poly_scale))])
    cv2.putText(image_np,'Vehicle Speed :', (550,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (102,255,102), 2)
    cv2.putText(image_np,'Collision Time :', (550,44), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(102,255,102), 2)
    cv2.putText(image_np,'Breaking Flag :', (550,66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (102,255,102), 2)
    return

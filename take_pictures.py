import time
import cv2

#file = "C:/Users/HP PRO/Desktop/picstaken/"
 
def tp(path):
    # Camera 0 is the integrated web cam on my netbook
    camera_port = 0
     
    #Number of frames to throw away while the camera adjusts to light levels
    ramp_frames = 30
     
    # Now we can initialize the camera capture object with the cv2.VideoCapture class.
    # All it needs is the index to a camera port.
    camera = cv2.VideoCapture(camera_port)
     
    # Captures a single image from the camera and returns it in PIL format
    #def get_image():
     # read is the easiest way to get a full image out of a VideoCapture object.

     
    # Ramp the camera - these frames will be discarded and are only used to allow v4l2
    # to adjust light levels, if necessary
    for i in xrange(ramp_frames):
        retval, im = camera.read()
        temp = im
    print("Taking image...")
    # Take the actual image we want to keep
    #camera_capture = get_image()

    file = path+"/"
    # A nice feature of the imwrite method is that it will automatically choose the
    # correct format based on the file extension you provide. Convenient!
    for i in range(1, 20):
        retval, im = camera.read()
        camera_capture = im
        cv2.imwrite(file+str(i)+'.png', camera_capture)
        time.sleep(.300)
     
    # You'll want to release the camera, otherwise you won't be able to create a new
    # capture object until your script exits
    del(camera)

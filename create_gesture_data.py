import os
import cv2
import glob
import json
import numpy as np
import tensorflow as tf

background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

file = open('checkpoints.json')
data = json.load(file)

test_path = data['paths'][0]['path_test_data']
path = data['paths'][0]['path_train_data']

traindata = glob.glob(path + "\*")
noOfGestures = len(traindata)
print("Number of Gestures already available: ", noOfGestures)

c_frame=-1
p_frame=-1

#Setting threshold for number of frames to compare
thresholdframes=50


## Let us restore the saved model
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('C:/Users/Raj Shah/Downloads/AHD_Project/handgest_1.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, 10))
y_test_images = np.zeros((1, 10))

# Real Time prediction
def predict(frame, y_test_images):
    image_size = 50
    num_channels = 3
    images = []
    image = frame
    cv2.imshow('test', image)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)

    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size, image_size, num_channels)

    ### Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [probabiliy_of gest0,......,probability_of_gest9]
    return np.array(result)


# Open Camera object
cap = cv2.VideoCapture(0)

# Decrease frame size (4=width,5=height)
cap.set(4, 700)
cap.set(5, 400)

h, s, v = 150, 150, 150
i = 0
while (i < 1000000):
    ret, frame = cap.read()

    cv2.rectangle(frame, (300, 300), (100, 100), (0, 255, 0), 0)
    crop_frame = frame[100:300, 100:300]
    # Blur the image
    # blur = cv2.blur(crop_frame,(3,3))
    blur = cv2.GaussianBlur(crop_frame, (3, 3), 0)

    # Convert to HSV color space
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))
    med = cv2.medianBlur(mask2, 5)

    ##Displaying frames
    cv2.imshow('main', frame)
    cv2.imshow('masked', med)

    ##resizing the image
    med = cv2.resize(med, (50, 50))
    ##Making it 3 channel
    med = np.stack((med,) * 3)
    ##adjusting rows,columns as per x
    med = np.rollaxis(med, axis=1, start=0)
    med = np.rollaxis(med, axis=2, start=0)
    ##Rotating and flipping correctly as per training image
    M = cv2.getRotationMatrix2D((25, 25), 270, 1)
    med = cv2.warpAffine(med, M, (50, 50))
    med = np.fliplr(med)
    ##converting expo to float
    np.set_printoptions(formatter={'float_kind': '{:f}'.format})
    ##printing index of max prob value
    ans = predict(med, y_test_images)
    # print(ans)
    # print(np.argmax(max(ans)))

    # Comparing for 50 continuous frames
    c_frame = np.argmax(max(ans))
    if (c_frame == p_frame):
        counter = counter + 1
        p_frame = c_frame
        if (counter == thresholdframes):
            print(ans)
            print("Gesture:" + str(c_frame))
            counter = 0
            i = 0
    else:
        p_frame = c_frame
        counter = 0

    # close the output video by pressing 'ESC'
    k = cv2.waitKey(2) & 0xFF
    if k == 27:
        break
    i = i + 1

cap.release()
cv2.destroyAllWindows()

"""
def cal_accum_avg(frame, accumulated_weight):
    global background

    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)


def segment_hand(frame, threshold=25):
    global background

    diff = cv2.absdiff(background.astype("uint8"), frame)

    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Grab the external contours for the image
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        return thresholded, hand_segment_max_cont


cam = cv2.VideoCapture(0)

num_frames = 0
element = 10
num_imgs_taken = 0
st = ''
res = []

def get_gesture_name(path):
    n = input("Enter the Gesture name: ")
    path = path + "\\" + n
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(test_path, name)):
        os.mkdir(os.path.join(test_path, name))
    return n

name = get_gesture_name(path)

while True:
    ret, frame = cam.read()

    # flipping the frame to prevent inverted image of captured frame...
    frame = cv2.flip(frame, 1)

    frame_copy = frame.copy()

    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

    # cv2.imshow("Sign Detection", frame_copy)
    # k = cv2.waitKey(0)
    # print(k, num_imgs_taken)

    # while k != 27 and num_imgs_taken < 1:
    #     k = cv2.waitKey(0)
    #     st = st + chr(k)
    #     cv2.putText(frame_copy, "Enter gesture name:" + st, (30, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 1)
    #
    # path = path + "\\" + st

    if num_frames < 60:
        cal_accum_avg(gray_frame, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255), 2)
            # cv2.imshow("Sign Detection",frame_copy)

    # Time to configure the hand specifically into the ROI...
    elif num_frames <= 299:

        hand = segment_hand(gray_frame)

        cv2.putText(frame_copy, "Adjust hand...Gesture for" + str(element), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        # Checking if hand is actually detected by counting number of contours detected...
        if hand is not None:
            thresholded, hand_segment = hand

            # Draw contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)

            # Also display the thresholded image
            cv2.imshow("Thresholded Hand Image", thresholded)

    else:

        # Segmenting the hand region...
        hand = segment_hand(gray_frame)

        # Checking if we are able to detect the hand...
        if hand is not None:

            # unpack the thresholded img and the max_contour...
            thresholded, hand_segment = hand

            # Drawing contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)

            cv2.putText(frame_copy, str(num_frames), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(frame_copy, str(num_frames)+"For" + str(element), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(frame_copy, str(num_imgs_taken + 1) + 'images For' + str(element), (200, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Displaying the thresholded image
            cv2.imshow("Thresholded Hand Image", thresholded)
            if num_imgs_taken <= 299:
                cv2.imwrite(path + "\\" + name + "\\gest" + str(noOfGestures) + "_" + str(num_imgs_taken + 1) + '.jpg',
                            thresholded)
            else:
                break
            num_imgs_taken += 1
        else:
            cv2.putText(frame_copy, 'No hand detected...', (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Drawing ROI on frame copy
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 3)

    cv2.putText(frame_copy, "DataFlair hand sign recognition_ _ _", (10, 20), cv2.FONT_ITALIC, 0.5, (51, 255, 51), 1)

    # increment the number of frames for tracking
    num_frames += 1

    # Display the frame with segmented hand
    cv2.imshow("Sign Detection", frame_copy)

    # Closing windows with Esc key...(any other key with ord can be used too.)
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

# Releasing camera & destroying all the windows...

cv2.destroyAllWindows()
cam.release()
"""
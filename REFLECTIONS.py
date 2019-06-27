from keras import backend as K
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import os
import glob
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import time
PADDING = 50
ready_to_detect_identity = True

import subprocess

def say(text):
    subprocess.call(['say', text])

FRmodel = faceRecoModel(input_shape=(3, 96, 96))

def triplet_loss(y_true, y_pred, alpha = 0.3):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

def prepare_database():
    database = {}

    # load all the images of individuals to recognize into the database
    for file_num in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file_num))[0]
        identity = ''.join([i for i in identity if not i.isdigit()])
        if identity in database:
            database[identity] = database[identity] + [img_path_to_encoding(file_num, FRmodel)]
        else:
            database[identity] = [img_path_to_encoding(file_num, FRmodel)]
    return database

def webcam_face_recognizer(database):
    """
    Runs a loop that extracts images from the computer's webcam and determines whether or not
    it contains the face of a person in our database.

    If it contains a face, an audio message will be played welcoming the user.
    If not, the program will process the next frame from the webcam
    """
    global ready_to_detect_identity

    cv2.namedWindow("MIRROR, MIRROR, ON THE WALL", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('MIRROR, MIRROR, ON THE WALL', 1000, 1200)
    vc = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    while vc.isOpened():
        _, frame = vc.read()
        img = frame

        # We do not want to detect a new identity while the program is in the process of identifying another person
        if ready_to_detect_identity:
            img = process_frame(img, frame, face_cascade)   
        
        key = cv2.waitKey(100)
        cv2.imshow("MIRROR, MIRROR, ON THE WALL", img)

        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("MIRROR, MIRROR, ON THE WALL")

def process_frame(img, frame, face_cascade):
    """
    Determine whether the current frame contains the faces of people from our database
    """
    global ready_to_detect_identity
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through all the faces detected and determine whether or not they are in the database
    identities = []
    for (x, y, w, h) in faces:
        x1 = x-PADDING
        y1 = y-PADDING
        x2 = x+w+PADDING
        y2 = y+h+PADDING

        img = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,0,0),2)

        identity = find_identity(frame, x1, y1, x2, y2)

        if identity is not None:
            identities.append(identity)

    if identities != []:
        cv2.imwrite('example.png',img)

        ready_to_detect_identity = False
        pool = Pool(processes=1) 
        # We run this as a separate process so that the camera feedback does not freeze
        pool.apply_async(welcome_users, [identities])
    return img

def find_identity(frame, x1, y1, x2, y2):
    """
    Determine whether the face contained within the bounding box exists in our database

    x1,y1_____________
    |                 |
    |                 |
    |_________________x2,y2

    """
    height, width, channels = frame.shape
    # The padding is necessary since the OpenCV face detector creates the bounding box around the face and not the head
    part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    
    return who_is_it(part_image, database, FRmodel)

def who_is_it(image, database, model):
    """
    Implements face recognition by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    encoding = img_to_encoding(image, model)
    
    min_dist = 100
    identity = None
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_encs) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dists = [np.linalg.norm(db_enc - encoding) for db_enc in db_encs]
        dist = np.mean(dists)
#        # print('distance for %s is %s' %(name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.55:
        return None
    else:
        return str(identity)

def welcome_users(identities):
    """ Outputs a welcome audio message to the users """
    global ready_to_detect_identity
    os.system('clear')
    print('Today is Thursday, June 26')
    welcome_message = 'Good morning %s.' % identities[0]
    print('Good morning %s' % identities[0])
    print('\n')
    say(welcome_message)
    say('Today is Thursday, june twenty six')

    if identities[0] == 'Abbi':
        # Meds
        print('\n')
        print('Today\'s Meds:')
        print('\tTake 200mg of Orilissa at: 10am, 4pm')
        say('Remember to take two hundred miligrams of Orilissa at ten a m and at four p m today.')
        time.sleep(1)

        # Appointments
        print('\n')
        print('Today\'s Appointments:')
        print('\t3:00 p.m. with Dr. Eddo')
        say('You have an appointment with doctor Eddo at three p m')
        time.sleep(1)

        # Refills
        print('\n')
        print('Today\'s Refills:')
        print('\tOrilissa')
        say('Remember to go to the pharmacy to refill your Orilissa')
    elif identities[0] == 'Vinny':
        # Meds
        print('\n')
        print('Today\'s Meds:')
        print('\tTake one 150mg injection of Skyrizi in the am')
        print('\tTake three Mavyret pills with your meals')
        say('Remember to take a hundred fifty milligram injection of skyrizi this morning And a Mavyret pill with each meal')
        time.sleep(1)

        # Appointments
        print('\n')
        print('Today\'s Appointments:')
        print('\t11:30 a.m. with Dr. Bone')
        say('You have an appointment with doctor Bone at eleven thirty a m')
        time.sleep(1)

        # Refills
        print('\n')
        print('Today\'s Refills:')
        print('\tMavyret')
        say('Remember to go to the pharmacy to refill your Mavyret')


    time.sleep(2)
    if identities[0] == 'Abbi':
        prompt = 'On a scale from one to ten what level of pain are your uterine fibers today?'
    elif identities[0] == 'Vinny':
        prompt = 'On a scale from one to ten how is your appetite today?'

    # Ask question
    say(prompt)
#    # Allow the program to start detecting identities again
    time.sleep(10)
    ready_to_detect_identity = True

if __name__ == "__main__":
    database = prepare_database()
    webcam_face_recognizer(database)

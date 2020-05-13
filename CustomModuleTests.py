import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import utilities
import DataAugmentation


train_images, train_keypoints = utilities.loadData('training.csv')
m = train_images.shape[0]
check_image = 20
aug = DataAugmentation.Augmentor()


def checkUtils():
    print("Training Images Shape:",train_images.shape)
    print("Training Keypoints Shape:", train_keypoints.shape)
	
    fig, axis = plt.subplots()
    utilities.plotSample(train_images[0], train_keypoints[0], axis, "check")
    



class ValidateAugmentor:
    '''
        Class containing methods to verify the data augmentation methods implemented in the Augmentor class in the Data                 Augmentation package
    '''
    
    def checkFlipAugmentation(self):
        #check horizontal flip
        H_flipped_images, H_flipped_keypoints = aug.flip(train_images, train_keypoints)

        fig, axis = plt.subplots(1,2)
        utilities.plotSample(H_flipped_images[check_image], H_flipped_keypoints[check_image], axis[0], "horizontally flipped image")
        utilities.plotSample(train_images[check_image], train_keypoints[check_image], axis[1], "true image")

        #check vertical flip
        V_flipped_images, V_flipped_keypoints = aug.flip(train_images, train_keypoints, False)

        fig, axis = plt.subplots(1,2)
        utilities.plotSample(V_flipped_images[check_image], V_flipped_keypoints[check_image], axis[0], "vertically flipped image")
        utilities.plotSample(train_images[check_image], train_keypoints[check_image], axis[1], "true image")


    def checkRotationAugmentation(self):
        rotation_angles = [15.0]
        rotated_images, rotated_keypoints = aug.rotate(train_images, train_keypoints, rotation_angles)
        #check that some images were not added due to boundary constraints
        print("Shape of the rotated images array:",rotated_images.shape)
        print("Shape of the original training images array:",train_images.shape)

        fig, axis = plt.subplots(1,3)
        utilities.plotSample(rotated_images[check_image], rotated_keypoints[check_image], axis[0], "clockwise")
        utilities.plotSample(train_images[check_image], train_keypoints[check_image], axis[1], "true image")
        utilities.plotSample(rotated_images[m+check_image], rotated_keypoints[m+check_image], axis[2], "counter clockwise")


    def checkBrightnessAugmentation(self):
        B = True
        D = True
        altered_images, altered_keypoints = aug.alterBrightness(train_images, train_keypoints, B, D, 1.2, 0.1)

        if B and D:
            fig, axis = plt.subplots(1,3)
        else:
            fig, axis = plt.subplots(1,2)

        if B and D:
            utilities.plotSample(altered_images[check_image], altered_keypoints[check_image], axis[0], "bright image")
            utilities.plotSample(train_images[check_image], train_keypoints[check_image], axis[1], "true image")
            utilities.plotSample(altered_images[m+check_image], altered_keypoints[m+check_image], axis[2], "damp")

        elif B:
            utilities.plotSample(altered_images[check_image], altered_keypoints[check_image], axis[0], "bright image")
            utilities.plotSample(train_images[check_image], train_keypoints[check_image], axis[1], "true image")

        else:
            utilities.plotSample(altered_images[check_image], altered_keypoints[check_image], axis[0], "damp image")
            utilities.plotSample(train_images[check_image], train_keypoints[check_image], axis[1], "true image")



    def checkShiftAugmentation(self):
        shifted_images, shifted_keypoints = aug.shiftImage(train_images, train_keypoints, [12])
        #check that some images were not added due to boundary constraints
        print("Shape of the shifted images array:",shifted_images.shape)
        print("Shape of the original training images array:",train_images.shape)

        fig, axis = plt.subplots()
        utilities.plotSample(shifted_images[check_image], shifted_keypoints[check_image], axis, "shifted image")



    def checkNoiseAugmentation(self):
        noisy_images = aug.addNoise(train_images, 0.1)
        noisy_keypoints = train_keypoints

        fig, axis = plt.subplots(1,2)
        utilities.plotSample(noisy_images[check_image], noisy_keypoints[check_image], axis[0], "noisy image")
        utilities.plotSample(train_images[check_image], train_keypoints[check_image], axis[1], "true image")

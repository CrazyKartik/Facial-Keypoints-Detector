import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle



class Augmentor:
    '''
        Class containing various image augmentation methods which can be used with grayscale images.
        Methods:
            flip
            rotate
            alterBrightness
            shiftImage
            addNoise
    '''
    
    def flip(self, images, keypoints, horizontal=True):
        '''
        Info-
            To flip the images horizontally or vertically and return the produced output

        Arguements-
            images: array, images to be flipped horizontally(shape = (no of images, rows, cols, channels))
            keypoints: array, position of the landmark for the provided images
            horizontal: boolean parameter, wether to flip horizontally or vertically, defaults to horizontal flip

        Returns-
            tuple containing flipped images and flipped keypoints
        '''

        flipped_keypoints = []
        if horizontal:
            #columnwise flip (horizontal flip)
            flipped_images = np.flip(images, axis=2)

            for points in keypoints:
              #change only X-coordinates and preserve the Y-coordinates
              flipped_keypoints.append([-p if i % 2 == 0 else p for i,p in enumerate(points)])

        else:
            #rowwise flip images(vertical flip)
            flipped_images = np.flip(images, axis=1)

            for points in keypoints:
              #change only Y-coordinates and preserve the X-coordinates
              flipped_keypoints.append([-p if i % 2 != 0 else p for i,p in enumerate(points)])

        flipped_keypoints = np.array(flipped_keypoints)

        return (flipped_images, flipped_keypoints)
    
    

    def rotate(self, images, keypoints, rotation_angles):
        '''
        Info-
            Can rotate the input images upto 30 degrees without losing the location of the keypoints
            2 images are produced per input image and rotation angle, 
              number of rotation angles is 2
              number of images is m,
              then the number of output images are (2 * m) * 2 - 2 images per image and angle

        Arguements-
            images: array, images to be rotated clockwise
            keypoints: array, position of the landmark for the provided images
            rotation_angles: list of floats, representing the rotation angles

        Returns-
          tuple containing images rotated both clockwise and counter clockwise and converted keypoints
        '''

        rotated_images = []
        rotated_keypoints = []

        images = images * 127.5 + 127.5
        keypoints = keypoints * 48.0 + 48.0

        for angle in rotation_angles:
            for ang in [-angle, angle]:
              #get the center point of the image around which the image is to be rotated (48,48) is the center of the images
              #positive angle is counter clockwise and negative angle is clockwise conventionally but in cv2, opposite is true

                M = cv2.getRotationMatrix2D((48,48), ang, 1.0) #1.0 is the scaling factor
                angle_rad = -ang * np.pi / 180.0

                for image,points in zip(images,keypoints):
                    img = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_AREA)
                    #flags/interpolation: to fill the missing data in the image (inter_cubic, inter_linear and inter_area are good choices here)

                    #subtract the value of the rotation point
                    rotated_keypoint = points - 48.
    	            
                    for i in range(0,len(rotated_keypoint),2):
                        rotated_keypoint[i] = rotated_keypoint[i] * np.cos(angle_rad) - rotated_keypoint[i+1] * np.sin(angle_rad)
                        rotated_keypoint[i+1] = rotated_keypoint[i] * np.sin(angle_rad) + rotated_keypoint[i+1] * np.cos(angle_rad)

                    #adding the previously subtracted value
                    rotated_keypoint += 48.
                    rotated_keypoint = np.array(rotated_keypoint)

                    if np.all(rotated_keypoint >= 0.0) and np.all(rotated_keypoint <= 96.0):
                        rotated_images.append(img)
                        rotated_keypoints.append(rotated_keypoint)


        rotated_images = np.array(rotated_images)
        rotated_images = rotated_images.reshape((-1,96,96,1))
        rotated_images = (rotated_images - 127.5) / 127.5
        rotated_keypoints = np.array(rotated_keypoints)
        rotated_keypoints = (rotated_keypoints - 48.) / 48.

        return (rotated_images, rotated_keypoints)
    
    
    
    
    def alterBrightness(self, images, keypoints, B=True, D=False, brighten=1.2, dampen=0.6):
        '''
        Info-
            To alter the brightness of the input images and return the results based on the preferred input

        Arguements-
            images: array, images to be brighten or dampen
            keypoints: array, landmark positions of the corresponding images
            B: boolean parameter, if needs brightness
            D: boolean parameter, if needs dampness
            brighten: float, factor by which to increase brightness, greater than 1
            dampen: float, factor by which to decrease brightness, less than 1

        returns-
            tuple of altered images and corresponding keypoints
        '''

        assert brighten > 1 and dampen < 1
        assert B or D

        altered_images = []
        images = images * 127.5 + 127.5

        if B:
            brightened_images = np.clip(images*brighten, -1.0, 1.0)
            altered_images.extend(brightened_images)
            
        if D:
            damped_images = np.clip(images*dampen, -1.0, 1.0)
            altered_images.extend(damped_images)
            
        altered_images = np.array(altered_images)
        altered_images = (altered_images - 127.5) / 127.5

        return altered_images, np.concatenate((keypoints,keypoints))
    
    
    
    
    def shiftImage(self, images, keypoints, shifts=[]):
        '''
        Info-
            To shift the images upwards, downwards, left and right to make the dominant features shift places and return the               shifted images

        Arguements:
            images: array, images to be shifted
            keypoints: array, keypoints to be converted
            shifts: list of positive ints, pixels by which to shift the image features, defaults to []

        Returns-
            tuple of the shifted images and the converted keypoints
        '''

        shifted_images = []
        shifted_keypoints = []

        keypoints = keypoints * 48. + 48.

        for shift in shifts:
            for (shift_x,shift_y) in [(shift,shift), (-shift,shift), (shift,-shift), (-shift,-shift)]:
                M = np.float32([
                              [1,0,shift_x],
                              [0,1,shift_y]
                            ])

                for image,keypoint in zip(images,keypoints):
                    shifted_image = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_AREA)
                    shifted_keypoint = np.array([point+shift_x if idx % 2 == 0 else point+shift_y for idx,point in enumerate(keypoint)])

                    if np.all(shifted_keypoint > 0.0) and np.all(shifted_keypoint < 96.0):
                        shifted_keypoints.append(shifted_keypoint)
                        shifted_images.append(shifted_image)

        shifted_images = np.array(shifted_images)
        shifted_images = shifted_images.reshape((-1,96,96,1))
        shifted_keypoints = np.array(shifted_keypoints)
        shifted_keypoints = (shifted_keypoints - 48.) / 48.

        return (shifted_images, shifted_keypoints)
    
    
    
    
    def addNoise(self, images, noise_scale=0.01):
        '''
        Info-
            To add random noise in the image

        Arguements:
            images: array, images in which noise is to be introduced
            noise_scale: float, scales down the added noise, defaults to 0.01

        Returns-
            array of noisy images
        '''

        noisy_images = []

        for image in images:
            noise = noise_scale*np.random.randn(96,96,1)
            noisy_image = noise + image
            noisy_image = np.clip(noisy_image, -1.0, 1.0)
            noisy_images.append(noisy_image)

        noisy_images = np.array(noisy_images)

        return noisy_images

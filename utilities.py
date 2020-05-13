import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle



def loadData(file=None, cols=None, drop_null=True):
    '''
    Info-
        load data from a csv file and remove missing data if needed
        
    Arguements-
        file: string, ony to be specified if training data is need else it loads testing data, defaults to None
        cols: list of string, columns to be extracted from the file, left blank if all columns are needed, defaults to None
        drop_null: boolean, specify if the missing is to be removed or not, defaults to True
        
    Returns-
    	tuple containing prepared training images and training labels
    '''

    if file:
        df = pd.read_csv('training.csv')
    
    else:
        df = pd.read_csv('test.csv')

    #Image column has pixel values separated by space in form of string
    df['Image'] = df['Image'].apply(lambda pix: np.fromstring(pix, sep=' '))

    if cols:
        df = df[list(cols) + 'Image']
    
    #fill missing values with previous ones
    if not drop_null:
        df = df.fillna(method='pad')

    #remove rows with missing values
    df = df.dropna()

    #Normalizing the input between 0 and 1
    X = (np.vstack(df['Image'].values) - 127.5) / 127.5
    X = X.astype(np.float32)

    #Only training.csv had the output labels, so load only when we need to train
    #Normalizing the output between -1 and 1 and shuffling the input data
    if file:
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48.0
        X, y = shuffle(X, y, random_state=1)
        y = y.astype(np.float32)

    else:
        y = None

    X = X.reshape((-1, 96, 96, 1))

    return X, y


def plotSample(image, keypoints, axis, title=None):
    '''
    Info-
        to display the image and plot the keypoints on the input image
        
    Arguements:
        image: array, image to be displayed
        keypoints: array, location of the keypoints on the image
        axis: plt object, graph on which the image and keypoints are to be plotted
        title: string, title for the graph, defaults to None
        
    Returns-
        None
    '''
    
    image = image.reshape((96,96))
    axis.imshow(image, cmap='gray')
    axis.scatter(keypoints[::2] * 48 + 48, keypoints[1::2] * 48 + 48, marker='o', s=10)
    if title:
        axis.title.set_text(title)
        
        
        
FIdLookup = 'IdLookupTable.csv'

IdLookup = pd.read_csv(FIdLookup)

def prepareSubmission(y_test,filename):
    '''
    Info-
        saves a .csv file that can be submitted to kaggle
    
    Arguements-
        y_test: pandas dataframe, predictions from the model
        filename: string, the name of the output file
    
    Returns-
    	None
    '''

    ImageId = IdLookup["ImageId"]
    FeatureName = IdLookup["FeatureName"]
    RowId = IdLookup["RowId"]
    
    submit = []
    for rowId,irow,landmark in zip(RowId,ImageId,FeatureName):
        submit.append([rowId, y_test[landmark].iloc[irow-1]])
    
    submit = pd.DataFrame(submit, columns=["RowId","Location"])
    
    # adjust the scale 
    submit["Location"] = submit["Location"]*48 + 48
    submit["Location"] = submit["Location"].clip(0,96)
    #check desired shape
    print(submit.shape)
    
    submit.to_csv(filename+".csv",index=False)
    print("Results stored to {}.csv.....".format(filename))

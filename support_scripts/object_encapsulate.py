import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from data.metadata import metaData
from config import *
from paths import *

# print(metaData)

class objectIndividual():
    """
    An abstraction which encapsulates the relation between elements of the 
    corresponding dataset. Every single instance of entity has its corresponding
    MNIST handwriting & speaker identifier as in the metadata artifact i.e.
    .data/metadata.py
    """

    def __init__( self, mnistLangFirst = 'Bangla', mnistLangFSecond = 'Bangla', dtype = np.float32 ):
        """
            Initialize the class object by loading corresponding 
            data objects from the dataset, whose objects / references are
            passed as arguments.
        """
        
        # initialize class object attributes with passed arguments
        self.mnistLangFirst = mnistLangFirst
        self.mnistLangFSecond = mnistLangFSecond
        self.dtype = dtype

        # Retrieve the Speech modality
        ( self.speech_X_train, self.speech_X_valid, self.speech_X_test, 
          self.speech_y_train, self.speech_y_valid, self.speech_y_test
        ) = self._getSpeechData()
        
        # Obtain the First MNIST Modality features & targets from the passed argument for train / test / valid
        ( self.mnistFirst_X_train, self.mnistFirst_X_valid, self.mnistFirst_X_test, 
          self.mnistFirst_y_train, self.mnistFirst_y_valid, self.mnistFirst_y_test
        ) = self._getMNISTData( self.mnistLangFirst )
        

        # Obtain the Second MNIST Modality features & targets from the passed argument for train / test / valid
        ( self.mnistSecond_X_train, self.mnistSecond_X_valid, self.mnistSecond_X_test, 
          self.mnistSecond_y_train, self.mnistSecond_y_valid, self.mnistSecond_y_test
        ) = self._getMNISTData( self.mnistLangFSecond )
    

    def _getStratSplit( self, X, y, splitRatio = 0.2 ):
        """
            Obtain the train / test split from the 
            Feature (X) & Target (y) variables passed
            as objects.

            The 'splitRatio' is the amount of data required
            to be allocated as 'test' data-set. 
        """

        # obtain the indices for the equivalent train / test splits
        trainIdx, valIdx = StratifiedShuffleSplit( n_splits = 1, test_size = splitRatio ).split(X, y).__next__()

        return X[trainIdx], X[valIdx], y[trainIdx], y[valIdx]


    def _loadTrainTest( self, npzFile ):
        """
            Load Train & Test data sets from passed
            .npz file as argument & return them as 
            container objects
        """
        
        # load Train / Test data from the .npz file passed as argument
        trainTestDict = dict( np.load( npzFile ) )

        # collate the loaded train / test dataset
        (X_Train, X_Test, y_Train, y_Test) = [ trainTestDict[k] for k in ["X_train", "X_test", "y_train", "y_test"] ]

        return (X_Train, X_Test, y_Train, y_Test)


    def _getSpeechData( self ):

        """
            This method retrieves speech data from .npz file & returns the 
            normalized versions of the Feature nd-arrays ( X_train, X_valid, X_test )
            & the target vectors ( y_train, y_valid, y_test ) 

            'speechDataDir' is loaded from paths.py which is imported as 'path'
            speechDataDir => directory where the speech data is stored
        """

        # List container for storing speaker's data
        X_train, X_test = [np.random.rand(0, 1, 13)] * 2
        y_train, y_test = [np.random.rand(0,)] * 2

        for name in ["jackson", "nicolas", "theo", "yweweler", "george", "lucas"]:

            # iterate over all the speaker names in the speech data  
            npz = os.path.join( speechDataDir, name + '_train_test.npz' ) # load the data from .npz file

            # load Train / Test data from the npz file
            ( xTrain, xTest, yTrain, yTest ) = self._loadTrainTest( npz )
            
            # append to the list container used to store the dataset
            X_train = np.append( X_train, xTrain, axis=0 )
            X_test = np.append( X_test, xTest, axis = 0 )
            y_train = np.append( y_train, yTrain, axis = 0 )
            y_test = np.append( y_test, yTest, axis = 0 )

        # Split the Train dataset into Train / Validation data
        (X_train, X_valid, y_train, y_valid) = self._getStratSplit( X_train, y_train )

        # remove unit dimensions from the ndarray
        X_train, X_valid, X_test = X_train.squeeze(), X_valid.squeeze(), X_test.squeeze()

        def normalize(x):
            """
                Normalize the data loaded from the .npz file
            """
            X_min = X_train.min(0)
            X_max = X_train.max(0)
            x = (x - X_min) / (X_max - X_min)
            
            return x

        # normalize the train / test/valid features
        X_train, X_valid, X_test = normalize( X_train.squeeze() ), normalize( X_valid.squeeze() ), normalize( X_test.squeeze() )

        # Type cast to appropriate data type
        ( X_train, X_valid, X_test, y_train, y_valid, y_test ) = ( X_train.astype(self.dtype), X_valid.astype(self.dtype), 
                                                                   X_test.astype(self.dtype), y_train.astype(self.dtype), 
                                                                   y_valid.astype(self.dtype), y_test.astype(self.dtype)
                                                                 )
        
        return ( X_train, X_valid, X_test, y_train, y_valid, y_test )


    def _getMNISTData( self, mnistLang ):

        """
            This function retrieves MNIST data from .npz file & returns the 
            normalized versions of the Feature nd arrays ( X_train, X_valid, X_test )
            & the target vectors ( y_train, y_valid, y_test )


            'mnistDataDir' is loaded from paths.py which is imported as 'path'
            mnistDataDir => directory where the MNIST data is stored
        """

        # Load train-test dataset from npz file
        npz = os.path.join(mnistDataDir, mnistLang + '_train_test.npz')
        
        
        (X_train, X_test, y_train, y_test) = self._loadTrainTest(npz)
        
        # Convert X to a 4-Dimensional Tensor which follows channel-last convention
        (X_train, X_test) = (X_train.reshape(-1, 28, 28, 1), X_test.reshape(-1, 28, 28, 1))
        
        # obtain the train / validation split dataset
        (X_train, X_valid, y_train, y_valid) = self._getStratSplit(X_train, y_train)
        
        # Type cast to appropriate data type / dtype
        (X_train, X_valid, X_test, y_train, y_valid, y_test) = ( X_train.astype(self.dtype), X_valid.astype(self.dtype),
                                                                 X_test.astype(self.dtype),  y_train.astype(self.dtype), 
                                                                 y_valid.astype(self.dtype), y_test.astype(self.dtype)
                                                               )

        def normalize(x):
            """
                Normalize the MNIST data
                -> dividing the data by 255.0 to bring it to 0 - 1 scale 
            """
            x = x / 255.0
            return x

        # normalize the Train / Test / Valid features
        X_train, X_valid, X_test = normalize(X_train), normalize(X_valid), normalize(X_test)

        return (X_train, X_valid, X_test, y_train, y_valid, y_test)

    
    def _getOneHot( self, y, classNumber ):
        """
            Returns one-hot encoded version of the specific class ('classNumber')
            'classNumber' -> int
        """
        return np.eye(classNumber)[y]


    def _sample( self, mnistFirst_X, mnistSecond_X, speech_X, mnistFirst_y, mnistSecond_y, speech_y ):
        """
           This method obtains a sample from the passed datasets as arguments.

        """

        def sampleY(y):
            """
                method to obtain samples for the specific label / Target data passed as argument 'y'
            """

            mnistFirst_X_y = mnistFirst_X[ mnistFirst_y == y ]
            mnistSecond_X_y = mnistSecond_X[ mnistSecond_y == y ]
            speech_X_y = speech_X[ speech_y == y ]
            
            # obtain maximum possible dimension size of MNIST (out of both first & second) so that no repetitions are necessary when sampled 
            size = mnistFirst_X_y.shape[0] if mnistFirst_X_y.shape[0] > mnistSecond_X_y.shape[0] else mnistSecond_X_y.shape[0]
            
            # Implementing a replacement policy
            if mnistFirst_X_y.shape[0] > mnistSecond_X_y.shape[0]:
                replace_1, replace_2 = (False, True)  # donot replace first but replace second
            
            elif mnistFirst_X_y.shape[0] < mnistSecond_X_y.shape[0]:
                replace_1, replace_2 = (True, False)  # donot replace second but replace first
            
            elif mnistFirst_X_y.shape[0] == mnistSecond_X_y.shape[0]:
                replace_1, replace_2 = (False, False) # replace neither


            # Initiate the replacement exercise => generate a random sample of length 'size' with the specific replacement policy for mnistFirst & mnistSecond
            mnistFirst_index = np.random.choice(mnistFirst_X_y.shape[0], size = size, replace = replace_1)
            
            mnistSecond_index = np.random.choice(mnistSecond_X_y.shape[0], size = size, replace = replace_2)
            
            speech_index = np.random.choice(speech_X_y.shape[0], size = size, replace = True)

            # mnistFirst sample modality
            mnistFirst_X_y = mnistFirst_X_y[mnistFirst_index]
            
            # mnistSecond sample modality
            mnistSecond_X_y = mnistSecond_X_y[mnistSecond_index]
            
            # speech modality
            speech_X_y = speech_X_y[speech_index]
            
            # Retrieve One-hot for the label Y
            label_Y = np.array( [ self._getOneHot( y, numDigit ) ] * size) # 'numDigit' loaded from config.py module imported here. numDigit = 10
            
            return (mnistFirst_X_y, mnistSecond_X_y, speech_X_y, label_Y)


        # initialize sample container objects
        mnistFirst = np.zeros( (0, 28, 28, 1) )
        mnistSecond = np.zeros( (0, 28, 28, 1) )
        speech = np.zeros( (0, 13) )
        label_y = np.zeros( (0, 10) )


        for label in range( numDigit ):

            # obtain sampled data for s specific target value
            mnistFirst_x, mnistSecond_x, speech_x, y = sampleY( label )
            
            # store the sampled data
            mnistFirst = np.append( mnistFirst, mnistFirst_x, axis = 0 )
            mnistSecond = np.append( mnistSecond, mnistSecond_x, axis = 0 )
            speech = np.append( speech, speech_x, axis = 0 )
            label_y = np.append( label_y, y, axis = 0 )

        
        indices = np.arange( label_y.shape[0] )
        # shuffle the data
        np.random.shuffle( indices )
        
        for ( firstMNIST, secondMNIST, spch, digitY ) in zip( mnistFirst[indices], mnistSecond[indices], speech[indices], label_y[indices] ):
            # return a generator for the tuple container of the dataset
            yield ( ( firstMNIST, secondMNIST, spch ) , digitY )

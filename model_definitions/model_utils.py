import torch
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from multimodal_variational_autoencoder import multimodal_variational_autoencoder


def model_save( model_object, best_model_flag, dir_location = './', filename = 'model.tar' ):
    """
        Save the model as a checkpoint while training in order to avoid
        information loss.
        The arguments passed:

            model_object : torch mmodel object undergoing training
            best_model_flag : Boolean flag variable whether the model is the best-possible one
            dir_location : directory location of the model
            filename : name of the .tar file to store the model
    """
    if not os.path.isdir( dir_location ):
    
        # if the directory does not exist then create one
        os.mkdir( dir_location )

    # save the model with the passed filename as argument
    torch.save( model_object, os.path.join( dir_location, filename ) )
    
    if best_model_flag:
        
        # if it is the best model so far, then save the model  by copying it to 'model_best.tar'
        shutil.copyfile( os.path.join( dir_location, filename ), os.path.join( dir_location, 'model_best.tar' ) )


def model_load( model_checkpoint_path, available_CUDA = False ):
    """
        Load a saved model tar file from a designated checkpoint passed as argument.
        Arguments:

            model_checkpoint_path : file location for the stored model
            available_CUDA : flag for using available CUDA cores

    """
    load_model_location = None
    if available_CUDA:
        # laod the model from the passed checkpoint location    
        load_model_location = torch.load( model_checkpoint_path ) 
    
    else:

        # load model from checkpoint while using 'storage' of CUDA Tensors from map_location
        load_model_location = torch.load( model_checkpoint_path, map_location = lambda stored, lod: stored )

    # instantiate model using loaded configs
    modelObject = multimodal_variational_autoencoder( load_model_location['n_latents'] )

    # load the model state
    modelObject.load_state_dict( load_model_location['state_dict'] )
    
    return modelObject


class loss_tracker( object ):
    """
        This class implements methods to keep track of and
        find the average value of the loss dduring training iterations
    """
    def __init__(self):
        """
            Initialise class object attirbutes
        """
        self.tracker_reset()

    def tracker_reset(self):
        """
            Re-initialise the loss tracker
        """
        self.loss = 0.0
        self.loss_average = 0.0
        self.loss_sum = 0.0
        self.num_iteration = 0.0

    def tracker_update( self, loss, iteration = 1 ):
        """
            Update the attributes of the loss tracker
            with successive iterations
        """
        self.loss = loss
        self.loss_sum += loss * iteration
        self.num_iteration += iteration
        self.loss_average = self.loss_sum / self.num_iteration


def returnMeanLogarithmVariance( modelObject, loadedDataObject, specificDigit, ifCUDA = False ):
    """
        This method returns the mean & logarithmic Variance of the data (images) from
        the loaded dataset.
    """
    modelObject.eval() # evaluate the passed model object

    firstImageData = None
    secondImageData = None
    speechData = None
    yLabel = None
    targetLabel = None

    for batchID, ( ( firstImage, secondImage, speech ), y ) in enumerate( loadedDataObject ):
        """
            iterate through the passed data & break
            once the target digit label is btained
        """

        firstImageData = firstImage
        secondImageData = secondImage
        speechData = speech
        yLabel = y

        targetLabel = np.argmax( y ).cpu().detach().numpy()

        if targetLabel == specificDigit :
            break
    
    if ifCUDA:
        """
            if CUDA cores are available then use them
        """
        firstImageData = firstImageData.cuda().float()
        secondImagedata = secondImageData.cuda().float()
        speechData = speechData.cuda().float()
        yLabel = yLabel.cuda().float()

    #print( targetLabel, specificDigit )

    firstImageData = torch.autograd.Variable( firstImageData ).float()
    secondImageData = torch.autograd.Variable( secondImageData ).float()
    speechData = torch.autograd.Variable( speechData ).float()
    yLabel = torch.autograd.Variable( yLabel ).float()

    # run Model Inference...
    (
        reconstructFirstImage, reconstructSecondImage, 
        reconstructSpeech, reconstructLabel, 
        mean, logarithmVariance
    ) = modelObject( firstImage = firstImageData, secondImage = secondImageData, speech = speechData )

    return mean, logarithmVariance


def displayImage( modelObject, tensorObject, firstImageFlag = True ):
    """
        Plot out the reconstructed Images from the passed Tensor Object
        by passing it through the Image Decoder in model architecture
    """
    reconstructImage = None

    if firstImageFlag:
        reconstructImage = modelObject.image1_decoder( tensorObject )

    else:
        reconstructImage = modelObject.image2_decoder( tensorObject )

    reconstructImage = reconstructImage.view(-1,28,28)
    fig = plt.figure( figsize = ( 20, 20 ) )
    
    dims = 20

    for i in range(0, dims * dims):

        imgNDArray = reconstructImage[ i , :, :].cpu().detach().numpy()
        
        # plot setup
        fig.add_subplot( dims, dims, i + 1 )
        plt.axis('off')
        # display the Image
        plt.subplots_adjust( wspace = 0, hspace = 0 )
        plt.imshow( imgNDArray,interpolation = 'nearest' )


def checkIfDisentangled( copyTensor, copyIndex, device, swap, limit = 100 ):
    """
        Analyze disentanglement in the latent space
        & return corresponding disentanglement tensor
    """
    
    dimTensor = torch.randn( limit, 512 ).to( device )
    lowerLimit = [ -swap * i for i in range( 1, limit // 2 + 1 ) ][ :: -1]
    upperLimit = [ swap * i for i in range( 1, limit // 2 + 1 ) ]

    # print( dimTensor.shape, copyTensor.shape )
    # print( len( lowerLimit + upperLimit ) )

    for idx, channel in enumerate( lowerLimit + upperLimit ):
        
        # print( '------> ', idx, channel )
        # print( dimTensor[idx].shape )
        lim = 512 / copyTensor.shape[0] # print( lim )
        dimTensor[ idx ] = deepcopy( torch.flatten( copyTensor[ ::, 0 : int(lim) ] ) )
        dimTensor[idx][copyIndex] += channel
    
    return torch.t( dimTensor )

def plotter( imageTensor, saveFileFlag = False, imageFileName = 'Generatedimage'):
    """
        ImageTensorplotter utility function
    """
    _, ax = plt.subplots( 1, 11, figsize=( 10,5 ) )
    
    anonymousFx = lambda x: x * 0.5 + 0.5
    imageTensor = anonymousFx( imageTensor.cpu() )
    
    for idx, axes in enumerate( ax.flatten() ):
        
        image = imageTensor[idx].detach().numpy()
        axes.imshow( image )
        axes.axis( 'off' )
    
    if saveFileFlag:
        plt.savefig( imageFileName )
    
    plt.show()

def intitiateReparmeterize( modelObject, loadedDataObject, digitLabel, cudaFlag = False ):
    """
        Reparameterize the modelObject passed with the dataset that has been loaded
    """
    mean, logVariance = returnMeanLogarithmVariance( modelObject, loadedDataObject, specificDigit = digitLabel, ifCUDA = cudaFlag )
    return modelObject.reparametrize( mean, logVariance )


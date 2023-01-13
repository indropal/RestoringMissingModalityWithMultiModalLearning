import numpy as np
import torch


def elboLoss( recon_image1 = None, image1 = None, recon_image2 = None, image2 = None, recon_speech = None,
              speech = None, recon_label = None, y = None, mu = None, logvar = None, lambda_image = 1.0,
              lambda_speech = 1.0, annealing_factor = 1
            ):
    """
        This function returns the ELBO loss for the specific modalities passed as argument.
        The argument parameters are as follows:

            recon_image1 : list of torch.Tensors / Variables for image1 modality
            recon_image2 : list of torch.Tensors / Variables for image2 modality
            recon_speech : list of torch.Tensors / Variables for speech modality
            recon_label : list of torch.Tensors / Variables for the label 'y' 

            image1 : list of torch.Tensors / variables whose sie matches recon_image1
            image2 : list of torch.Tensors / Variables whose size matches recon_image2
            speech : list of torch.Tensors / Variables whose size matches recon_speech
            y : label for the passed set image1, image2, speech
            
            mu : a torch.Tensor representing the mean of the variational ditribution
            logvar: a torch.Tensor representing the logarithm of the variational distribution
            
            lambda_image : float value which is a weight for image Binary Cross Entropy (BCE)
            lambda_speech : float value which is a weight for speech Binary Cross Entropy (BCE)
            annealing_factor : float value which is the weight assigned to the Kullback-Leiber regularizer
    """
    
    batch_size   = mu.size(0) # retrieve the size of the batch

    image1_bce, image2_bce, speech_mse, label_ce  = 0, 0, 0, 0 # initialize the cost variables
    
    if recon_image1 is not None and image1 is not None:
        # case when 'recon_image1' and 'image1' are passed as arguments to the function
        image1_bce = torch.mean( torch.sum( torch.nn.BCEWithLogitsLoss( reduction = 'none' )( recon_image1.view(-1, 1 * 28 * 28), 
                                                                                              image1.view(-1, 1 * 28 * 28)
                                                                                            ),
                                            dim = 1
                                          ),
                                 dim = 0
                               )
    
        #print("-------> Reconstruction of Image1: ", image1_bce)

    if recon_image2 is not None and image2 is not None:
        # case when 'recon_image2' and 'image2' are passed as arguments to the function
        image2_bce= torch.mean( torch.sum( torch.nn.BCEWithLogitsLoss( reduction = 'none' )( recon_image2.view(-1, 1 * 28 * 28), 
                                                                                             image2.view(-1, 1 * 28 * 28)
                                                                                           ),
                                           dim = 1
                                         ),
                                dim = 0
                              )

        #print("--------> Reconstruction of image2: ", image2_bce)
        
    if recon_speech is not None and speech is not None:
        # case when attribute 'speech' is passed as argument to the function
        speech_mse = torch.mean( torch.sum( torch.nn.MSELoss( reduction='none' )( recon_speech, speech ),
                                                              dim = 1
                                          ),
                                 dim = 0
                               )
        #print("--------> Reconstruction of speech attribute: ", speech_mse)
    
    if recon_label is not None and y is not None:
        # case when the label 'y' is passed as argument to the function
        label_ce = torch.mean( torch.nn.CrossEntropyLoss( reduction = 'none' )( recon_label, torch.argmax( y, dim = 1 )
                                                                              ),
                               dim = 0
                             )
        #print('----------> Classifier label: ', label_ce)

    KLD = -0.5 * torch.mean( torch.sum( 1 + logvar - mu.pow(2) - logvar.exp(), dim = 1 ), dim = 0 )

    ELBO_loss = (lambda_image * image1_bce) + (lambda_image * image2_bce) + (lambda_speech * speech_mse) + label_ce + annealing_factor * KLD
    
    # print("Kullback Leiber Divergence: {} | Computed ELBO loss: {}".format( KLD, ELBO_loss )  )
    
    return ELBO_loss, image1_bce, image2_bce, speech_mse, label_ce
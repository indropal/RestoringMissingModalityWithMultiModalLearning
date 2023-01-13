import torch
from torch.autograd import Variable

from image_encoder_decoder import image_encoder, image_decoder, Swish
from speech_encoder_decoder import speech_encoder, speech_decoder
from experts import prior_expert, product_of_experts


class Swish( torch.nn.Module ):
    """
        Reference Article for Swish activation function: https://arxiv.org/abs/1710.05941
    """
    def forward(self, feedTensor):
        return feedTensor * ( torch.tanh( torch.nn.functional.softplus( feedTensor ) ) )


class label_decoder( torch.nn.Module ):
    """
        Define the label_decoder class with parameter:
            'nLatents' : which denotes the number of latent dimensions 
    """

    def __init__( self, nLatents ):
        super( label_decoder, self).__init__()

        # instantiate the architecture of the decoder
        self.fully_connect_1   = torch.nn.Linear( nLatents, 512 )
        self.fully_connect_2   = torch.nn.Linear( 512, 512 )
        self.fully_connect_3   = torch.nn.Linear( 512, 512 )
        self.fully_connect_4   = torch.nn.Linear( 512, 10 )

        # instantiate the Swish activation
        self.swish = Swish()

    def forward(self, inputTensor):
        """
            Define the forward propagation method for decoder
        """
        interimTensor = self.swish( self.fully_connect_1( inputTensor ) )
        interimTensor = self.swish( self.fully_connect_2( interimTensor ) )
        interimTensor = self.swish( self.fully_connect_3( interimTensor ) )

        # return the ouput Tensor from the final Layer
        return self.fully_connect_4( interimTensor )


class data_iterate( torch.utils.data.IterableDataset ):
    """
        Custom class definition to iterate / parse through the
        entire dataset.

    """
    def __init__( self, data, data_iterate_config = 'train' ):
        """
            Initialize the dataset iterator object & its attributes
            with the attirbutes of the passed data object
        """
        self.dataset = data
        self.data_iterate_config = data_iterate_config


    def sample_data( self ):
        """
            iterate through the entire dataset to obtain a sample of
            the data - for all components of dataset i.e. train / test / validation
        """
        if self.data_iterate_config == 'train':
            
            # iterate through 'Train' data to obtain sample of the train-dataset
            for data_sample in self.dataset._sample( self.dataset.mnistFirst_X_train, self.dataset.mnistSecond_X_train, self.dataset.speech_X_train, self.dataset.mnistFirst_y_train, self.dataset.mnistSecond_y_train, self.dataset.speech_y_train ):
                yield data_sample

        elif self.data_iterate_config == 'test':
            
            # iterate through 'Test' data to obtain sample of the test-dataset
            for data_sample in self.dataset._sample( self.dataset.mnistFirst_X_test, self.dataset.mnistSecond_X_test, self.dataset.speech_X_test, self.dataset.mnistFirst_y_test, self.dataset.mnistSecond_y_test, self.dataset.speech_y_test ):
                yield data_sample

        else:
            
            # iterate through 'Validation' data to obtain sample of the validation-dataset
            for data_sample in self.dataset._sample( self.dataset.mnistFirst_X_valid, self.dataset.mnistSecond_X_valid, self.dataset.speech_X_valid, self.dataset.mnistFirst_y_valid, self.dataset.mnistSecond_y_valid, self.dataset.speech_y_valid ):
                yield data_sample

    def __iter__( self ):
        """
            Overriding iteration method of Iterabledataset module
        """
        return self.sample_data()


class multimodal_variational_autoencoder( torch.nn.Module ):
    """
        Defintion of Multimodal Variational Autoencoder architecture,
        methods & object attributes to provide functionalities to
        the model.
        
        Arguments passed:
            nLatent: the no. of latent dimensions
    """

    def __init__(self, nLatent):
        """
            Initialize the Autoencoder object with its various attributes
            and other companion object instances
        """
        super( multimodal_variational_autoencoder, self ).__init__()

        # instantiate image_encoder / image_decoder objects for first set of images
        self.image1_encoder = image_encoder(nLatent)
        self.image1_decoder = image_decoder(nLatent)

        # instantiate image_encoder / image_decoder objects for second set of images
        self.image2_encoder = image_encoder(nLatent)
        self.image2_decoder = image_decoder(nLatent)

        # instantiate the speech encoder / speech_decoder objects for speech
        self.speech_encoder  = speech_encoder(nLatent)
        self.speech_decoder  = speech_decoder(nLatent)

        # instantiate label_decoder object
        self.label_decoder = label_decoder(nLatent)

        # instantiate product_of_experts object
        self.experts = product_of_experts()

        # initialise number of latent dimensions
        self.nLatent = nLatent

        # check the dtype
        self.float()

    def weights_init( self, init_layer_Tensor ):
        """
            Initialize weight & bias of the layer with zeros
        """
        if isinstance( init_layer_Tensor, torch.nn.Linear ):

            # initialise if the input layer tensor is that of a Linear Layer
            torch.nn.init.zeros_( init_layer_Tensor.weight )
            torch.nn.init.zeros_( init_layer_Tensor.bias )


    def reparametrize( self, mean, logarithm_variance ):
        """
            Return the re-initialized variables for
            specific model attributes
        """
        if self.training:

            # if under training phase
            std_dev = logarithm_variance.mul( 0.5 ).exp_()
            epsilon = Variable( std_dev.data.new( std_dev.size() ).normal_() )

            return epsilon.mul( std_dev ).add_( mean )
        
        else:
            # if not under training
          return mean


    def forward( self, firstImage = None, secondImage = None, speech = None ):
        """
            Redefine forward propagation method 
        """
        mean, logarithm_variance = self.deduce_gaussian( firstImage, secondImage, speech)
        
        # Reparameterize the sample
        reparam = self.reparametrize( mean, logarithm_variance )
        
        # Gaussian reconstruction of the autoencoder attributes via defined Decoders 
        reconstruct_image_first = self.image1_decoder( reparam )
        reconstruct_image_second = self.image2_decoder( reparam )
        reconstruct_speech = self.speech_decoder( reparam )
        reconstruct_label = self.label_decoder( reparam )
        
        # return reconstructed params & gaussian arguments
        return ( reconstruct_image_first, reconstruct_image_second,
                 reconstruct_speech, reconstruct_label,
                 mean, logarithm_variance
               )


    def deduce_gaussian( self, firstImage = None, secondImage = None, speech = None ):

        batch_size = 0    
        # initialize the batch size
        if firstImage is not None:
            batch_size = firstImage.size( 0 )
        
        elif secondImage is not None:
            batch_size = secondImage.size( 0 )
        
        else : 
            batch_size = speech.size( 0 ) 
        
        # check for CUDA availability
        cuda_available = next( self.parameters() ).is_cuda
        
        # initialize Gussian distribution parameters from prior_expert
        dimension = ( 1, batch_size, self.nLatent )
        mean, logarithm_variance = prior_expert( dimension, initiate_CUDA = cuda_available )

        if  firstImage  is not None:
            firstImage_mean, firstImage_logarithm_variance = self.image1_encoder( firstImage )
            # concatenate mean
            mean = torch.cat(( mean, firstImage_mean.unsqueeze( 0 ) ), dim = 0 )
            # concatenate logarithm variance
            logarithm_variance = torch.cat(( logarithm_variance, firstImage_logarithm_variance.unsqueeze( 0 ) ), dim = 0 )

        if  secondImage  is not None:
            secondImage_mean, secondImage_logarithm_variance = self.image2_encoder( secondImage )
            # concatenate mean
            mean = torch.cat( ( mean, secondImage_mean.unsqueeze( 0 ) ), dim = 0 )
            # concatenate logarithm variance
            logarithm_variance = torch.cat( ( logarithm_variance, secondImage_logarithm_variance.unsqueeze( 0 ) ), dim = 0 )

        if speech is not None:
            speech_mean, speech_logarithm_variance = self.speech_encoder( speech )
            # concatenate mean
            mean = torch.cat( ( mean, speech_mean.unsqueeze(0) ), dim = 0 )
            # concatenate logarithm variance
            logarithm_variance = torch.cat( ( logarithm_variance, speech_logarithm_variance.unsqueeze(0) ), dim = 0 )

        # product of experts to combine gaussians
        mean, logarithm_variance = self.experts( mean, logarithm_variance )
        
        # return resultant Product of Experts Distibution 
        return mean.float(), logarithm_variance.float()

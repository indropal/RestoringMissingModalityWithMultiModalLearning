import torch


class Swish( torch.nn.Module ):
    """
        Reference Article for Swish activation function: https://arxiv.org/abs/1710.05941
    """
    def forward(self, feedTensor):
        return feedTensor * ( torch.tanh( torch.nn.functional.softplus( feedTensor ) ) )


class image_encoder( torch.nn.Module ):
    """
        Define the image_encoder class with parameter:
            'nLatents' : which denotes the number of latent dimensions 
    """
    def __init__( self, nLatents ):
        super( image_encoder, self ).__init__()

        # Create the encoder architecture
        self.fully_connect_1 = torch.nn.Linear( 784, 2048 )
        self.fully_connect_1_2 = torch.nn.Linear( 2048, 1024 )
        self.fully_connect_2 = torch.nn.Linear( 1024, 512 )
        self.fully_connect_3_1 = torch.nn.Linear( 512, nLatents )
        self.fully_connect_3_2 = torch.nn.Linear( 512, nLatents ).apply( self.weights_init )

        # check the dtype
        self.float()
                
        # instantiate the Swish activation
        self.swish = Swish()

    def weights_init( self, layer ):
        # initialize the weights with zeros
        torch.nn.init.zeros_( layer.weight )
        torch.nn.init.zeros_( layer.bias )

    def forward( self, inputTensor ):
        """
            Define the forward propagation method for encoder
        """
        processedInputTensor = inputTensor.view( -1, 784 )
        interimTensor = self.fully_connect_1( processedInputTensor.float() )
        interimTensor = self.swish( interimTensor )
        interimTensor = self.swish( self.fully_connect_1_2( interimTensor ) )
        interimTensor = self.swish( self.fully_connect_2( interimTensor ) )

        # return final layer Tensors
        return self.fully_connect_3_1( interimTensor ), self.fully_connect_3_2( interimTensor )



class image_decoder( torch.nn.Module ):
    """
        Define the Image Decoder with parameter:
            'nLatents' : number of Latent dimensions
    """

    def __init__( self, nLatents ):
        super( image_decoder, self).__init__()

        # create the decoder architecture
        self.fully_connect_1 = torch.nn.Linear( nLatents, 512 )
        self.fully_connect_2 = torch.nn.Linear( 512, 512 )
        self.fully_connect_3 = torch.nn.Linear( 512, 512 )
        self.fully_connect_3_4 = torch.nn.Linear( 512, 512 )
        self.fully_connect_4 = torch.nn.Linear( 512, 784 )

        # check the dtype
        self.float()

        # instantiate the Swish activation
        self.swish = Swish()

    def forward( self, inputTensor ):
        """
            Define the forward propagation method for the decoder
        """
        processedInputeTensor = self.swish( self.fully_connect_1( inputTensor ) )
        interimTensor = self.swish( self.fully_connect_2( processedInputeTensor ) )
        interimTensor = self.swish( self.fully_connect_3( interimTensor ) )
        interimTensor = self.swish( self.fully_connect_3_4( interimTensor ) )

        # return the ouput Tensor
        return self.fully_connect_4( interimTensor )
import torch


class Swish( torch.nn.Module ):
    """
        Reference Article for Swish activation function: https://arxiv.org/abs/1710.05941
    """
    def forward(self, feedTensor):
        return feedTensor * ( torch.tanh( torch.nn.functional.softplus( feedTensor ) ) )


class speech_encoder( torch.nn.Module ):
    """
        Define the speech_encoder class with parameter:
            'nLatents' : which denotes the number of latent dimensions 
    """
    def __init__(self, nLatents):

        super( speech_encoder, self).__init__()

        # instantiate the speech-encoder architecture
        self.fully_connect_1 = torch.nn.Linear(13, 512)
        self.fully_connect_2 = torch.nn.Linear(512, 512)
        self.fully_connect_3_1 = torch.nn.Linear(512, nLatents)
        self.fully_connect_3_2 = torch.nn.Linear(512, nLatents).apply( self.weights_init )

        # instantiate the Swish activation
        self.swish = Swish()

    def weights_init( self, layer ):
        # initialize the weights with zeros
        torch.nn.init.zeros_( layer.weight )
        torch.nn.init.zeros_( layer.bias )
    
    def forward(self, inputTensor):
        """
            Define the forward propagation method for encoder
        """        
        interimTensor = self.swish( self.fully_connect_1( inputTensor.float() ) )
        interimTensor = self.swish( self.fully_connect_2( interimTensor ) )

        # return final layer Tensors
        return self.fully_connect_3_1( interimTensor ), self.fully_connect_3_2( interimTensor )



class speech_decoder( torch.nn.Module ):
    """
        Define the speech_decoder class with parameter:
            'nLatents' : which denotes the number of latent dimensions 
    """
    def __init__(self, n_latents):
        super( speech_decoder, self ).__init__()

        # instantiate the speech-decoder architecture
        self.fully_connect_1 = torch.nn.Linear(n_latents, 512)
        self.fully_connect_2 = torch.nn.Linear(512, 512)
        self.fully_connect_3 = torch.nn.Linear(512, 512)
        self.fully_connect_4 = torch.nn.Linear(512, 13)

        # instantiate the Swish activation
        self.swish = Swish()


    def forward(self, inputTensor):
        """
            Define the forward propagation method for decoder
        """        
        interimTensor = self.swish( self.fully_connect_1( inputTensor ) )
        interimTensor = self.swish( self.fully_connect_2( interimTensor ) )
        interimTensor = self.swish( self.fully_connect_3( interimTensor ) )
        
        # return final layer Tensor
        return self.fully_connect_4( interimTensor )
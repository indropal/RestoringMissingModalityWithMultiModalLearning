import torch


def prior_expert( dimension, initiate_CUDA = False ):
    """
        Special case of gaussian distribution, Spherical Gaussian: N(0, 1).
        Arguments:
            dimension: dimension of Gaussian distribution
           initiate_cuda: use CUDA for compute
    """
    
    mean = torch.autograd.Variable( torch.zeros( dimension ) )
    log_variance = torch.autograd.Variable( torch.zeros( dimension ) )
    
    # if initiate_CUDA:
    #     # make use of available CUDA cores
    #     mean = mean.cuda()
    #     log_variance = log_variance.cuda()

    # return the computed params
    return mean.float(), log_variance.float()


class product_of_experts( torch.nn.Module ):
    """
        Implementing Generalized Product of Experts:
                https://arxiv.org/pdf/1410.7827.pdf

        Return product of experts params
        Arguments passed: 
        
            mean_experts : mean of experts
            log_variance_experts : logarithm variance of experts
    """

    def forward( self, mean_experts, log_variance_experts, epsilon = 1e-8 ):
        """
            Computing params for product of experts
            Arguments:

                mean_experts : Mean of Distribution of experts
                log_variance_experts : Logarithm of variance of experts
                eps : constant term to avoid Divide By Zero error and Logarithm of Zero
        """

        variance = torch.exp( log_variance_experts ) + epsilon
        
        # T-term / precision of Gaussian expert (i-th) 
        T = 1.0 / ( variance + epsilon )
        
        product_of_experts_mean = torch.sum( mean_experts * T, dim = 0 ) / torch.sum( T, dim = 0 )

        # generating parameters for Product of Experts
        product_of_experts_variance = 1.0 / torch.sum( T, dim = 0 )
        product_of_experts_log_variance = torch.log( product_of_experts_variance + epsilon )
        
        return product_of_experts_mean.float(), product_of_experts_log_variance.float()

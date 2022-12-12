import numpy as np
from elbo_loss import elboLoss
import torch

def initReport():
    """ 
        Initialize dictionary of history lists containing details of each modality 
        as well as for each combination of image & metrics and the total loss.
    """
    
    report = {

        "m1": {
            'elbo': [],
            'image1_bce': [],
            'image1_rmse': [],
            'image2_bce': [],
            'image2_rmse': [],
            'speech_mse': [],
            'speech_rmse': [],
            'label_ce': [],
            'label_acc': []
        },

        "m2": {
            'elbo': [],
            'image1_bce': [],
            'image1_rmse': [],
            'image2_bce': [],
            'image2_rmse': [],
            'speech_mse': [],
            'speech_rmse': [],
            'label_ce': [],
            'label_acc': []
        },

        "m3": {
            'elbo': [],
            'image1_bce': [],
            'image1_rmse': [],
            'image2_bce': [],
            'image2_rmse': [],
            'speech_mse': [],
            'speech_rmse': [],
            'label_ce': [],
            'label_acc': []
        },

        "m1m2": {
            'elbo': [],
            'image1_bce': [],
            'image1_rmse': [],
            'image2_bce': [],
            'image2_rmse': [],
            'speech_mse': [],
            'speech_rmse': [],
            'label_ce': [],
            'label_acc': []
        },

        "m2m3": {
            'elbo': [],
            'image1_bce': [],
            'image1_rmse': [],
            'image2_bce': [],
            'image2_rmse': [],
            'speech_mse': [],
            'speech_rmse': [],
            'label_ce': [],
            'label_acc': []
        },

        "m1m3": {
            'elbo': [],
            'image1_bce': [],
            'image1_rmse': [],
            'image2_bce': [],
            'image2_rmse': [],
            'speech_mse': [],
            'speech_rmse': [],
            'label_ce': [],
            'label_acc': []
        },

        "m1m2m3": {
            'elbo': [],
            'image1_bce': [],
            'image1_rmse': [],
            'image2_bce': [],
            'image2_rmse': [],
            'speech_mse': [],
            'speech_rmse': [],
            'label_ce': [],
            'label_acc': []
        },

        "total_loss": []
    }

    return report

def meanMetricsReportGen( history, epochHistory ):
    """
        Take the mean of each metric in dict 'history' & store it in 'epochHistory' as the 
        summarized epoch value for the metric
    """
    for combo in ["m1", "m2", "m3", "m1m2", "m2m3", "m1m3", "m1m2m3"]:
        for metric in ["elbo", "image1_bce", "image1_rmse", "image2_bce", "image2_rmse", 
                      "speech_mse", "speech_rmse", "label_ce", "label_acc"]:
            
            try:
                epochHistory[ combo ][ metric ].append( torch.mean( torch.stack( history[ combo ][ metric ] ) ).item() )
            except TypeError:
                epochHistory[ combo ][ metric ].append( np.mean( history[ combo ][ metric ] ).item() )
    
    
    epochHistory["total_loss"].append( torch.mean( torch.stack( history["total_loss"] ) ).item() )
    
    return epochHistory

def reportGen( model, history, image1, image2, speech, y, batch_idx, annealing_factor, lambda_image, lambda_speech ):
    """
        Tabulate the performance details of the model to store it in the initialized metrics dictionary.
    """

    def _acc(y_true, y_pred):
        """
            Efficient way for finding accuracy in the Training Batch 
            Reference to StackOverflow Article : https://stackoverflow.com/a/44130997
        """
        
        return ( torch.argmax( y_true, dim = 1 ) == torch.argmax( y_pred, dim = 1) ).sum() / float( len(y_true) )

    def _rmse(y_true, y_pred):
        """
            Return the RMSE for the Training Batch
        """

        return torch.sqrt( torch.mean( ( y_pred - y_true )**2 ) )

    total_loss = 0
    
    # find the ELBO Loss by using the entire data passed to the model => Both images & speech
    recon_image1, recon_image2, recon_speech, recon_label, mu, logvar = model( image1, image2, speech )

    elbo_Loss, image1_bce, image2_bce, speech_mse, label_ce = elboLoss( recon_image1, image1, recon_image2, image2,
                                                                        recon_speech, speech, recon_label, y , 
                                                                        mu, logvar, lambda_image = lambda_image, 
                                                                        lambda_speech = lambda_speech,
                                                                        annealing_factor = annealing_factor
                                                                      )
    total_loss += elbo_Loss
    history['m1m2m3']['elbo'].append( elbo_Loss )
    history['m1m2m3']['image1_bce'].append( image1_bce )
    history['m1m2m3']['image1_rmse'].append( _rmse( image1.view( -1, 1 * 28 * 28 ), recon_image1.view( -1, 1 * 28 * 28 ) ) )
    history['m1m2m3']['image2_bce'].append( image2_bce )
    history['m1m2m3']['image2_rmse'].append( _rmse( image2.view( -1, 1 * 28 * 28 ), recon_image2.view( -1, 1 * 28 * 28 ) ) )
    history['m1m2m3']['speech_mse'].append( speech_mse )
    history['m1m2m3']['speech_rmse'].append( _rmse( speech, recon_speech ) )
    history['m1m2m3']['label_ce'].append( label_ce )
    history['m1m2m3']['label_acc'].append( _acc( y, recon_label ) )


    # find the ELBO Loss by using only one image i.e. single image data -> image1 ONLY & not image2
    recon_image1, _ , _ , recon_label, mu, logvar = model( firstImage = image1 )

    elbo_Loss, image1_bce, image2_bce, speech_mse, label_ce = elboLoss( recon_image1 = recon_image1, image1 = image1,
                                                                        recon_label = recon_label, y = y , mu = mu,
                                                                        logvar = logvar, lambda_image = lambda_image,
                                                                        lambda_speech = lambda_speech,
                                                                        annealing_factor = annealing_factor
                                                                      )
    total_loss += elbo_Loss
    history['m1']['elbo'].append( elbo_Loss )
    history['m1']['image1_bce'].append( image1_bce )
    history['m1']['image1_rmse'].append( _rmse( image1.view( -1, 1 * 28 * 28 ), recon_image1.view( -1, 1 * 28 * 28 ) ) )
    history['m1']['image2_bce'].append( image2_bce )
    history['m1']['image2_rmse'].append( _rmse( image2.view( -1, 1 * 28 * 28 ), recon_image2.view( -1, 1 * 28 * 28 ) ) )
    history['m1']['speech_mse'].append( speech_mse )
    history['m1']['speech_rmse'].append( _rmse( speech, recon_speech ) )
    history['m1']['label_ce'].append( label_ce )
    history['m1']['label_acc'].append( _acc( y, recon_label ) )



    # find the ELBO Loss by using only one image i.e. single image data -> image 2 ONLY & not image1
    _ , recon_image2, _,recon_label, mu, logvar = model( secondImage = image2 )

    elbo_Loss, image1_bce, image2_bce, speech_mse, label_ce = elboLoss( recon_image2 = recon_image2, image2 = image2,
                                                                        recon_label = recon_label, y = y , mu = mu,
                                                                        logvar = logvar, lambda_image = lambda_image,
                                                                        lambda_speech = lambda_speech,
                                                                        annealing_factor = annealing_factor
                                                                      )
    total_loss += elbo_Loss
    history['m2']['elbo'].append( elbo_Loss )
    history['m2']['image1_bce'].append( image1_bce )
    history['m2']['image1_rmse'].append( _rmse( image1.view( -1, 1 * 28 * 28 ), recon_image1.view( -1, 1 * 28 * 28 ) ) )
    history['m2']['image2_bce'].append( image2_bce )
    history['m2']['image2_rmse'].append( _rmse( image2.view( -1, 1 * 28 * 28 ), recon_image2.view( -1, 1 * 28 * 28 ) ) )
    history['m2']['speech_mse'].append( speech_mse )
    history['m2']['speech_rmse'].append( _rmse( speech, recon_speech ) )
    history['m2']['label_ce'].append( label_ce )
    history['m2']['label_acc'].append( _acc( y, recon_label ) )


    # find the ELBO Loss by using only Speech data i.e. none of the image data -> DO NOT USE image1 & image2
    _ , _ , recon_speech, recon_label, mu, logvar = model( speech = speech )

    elbo_Loss, image1_bce, image2_bce, speech_mse, label_ce = elboLoss( recon_speech = recon_speech, speech = speech, 
                                                                        recon_label = recon_label, y = y, mu = mu,
                                                                        logvar = logvar, lambda_image = lambda_image,
                                                                        lambda_speech = lambda_speech,
                                                                        annealing_factor = annealing_factor
                                                                      )
    total_loss += elbo_Loss
    history['m3']['elbo'].append( elbo_Loss )
    history['m3']['image1_bce'].append( image1_bce )
    history['m3']['image1_rmse'].append( _rmse( image1.view( -1, 1 * 28 * 28 ), recon_image1.view( -1, 1 * 28 * 28 ) ) )
    history['m3']['image2_bce'].append( image2_bce )
    history['m3']['image2_rmse'].append( _rmse( image2.view( -1, 1 * 28 * 28 ), recon_image2.view( -1, 1 * 28 * 28 ) ) )
    history['m3']['speech_mse'].append( speech_mse )
    history['m3']['speech_rmse'].append( _rmse( speech, recon_speech ) )
    history['m3']['label_ce'].append( label_ce )
    history['m3']['label_acc'].append( _acc( y, recon_label ) )
    

    #  find the ELBO Loss by using both image data -> USE image1 & image2
    recon_image1, recon_image2, _, recon_label, mu, logvar = model( firstImage = image1, secondImage = image2 )

    elbo_Loss, image1_bce, image2_bce, speech_mse, label_ce = elboLoss( recon_image1 = recon_image1, image1 = image1,
                                                                        recon_image2 = recon_image2, image2 = image2,
                                                                        recon_label = recon_label, y = y , mu = mu,
                                                                        logvar = logvar, lambda_image = lambda_image,
                                                                        lambda_speech = lambda_speech,
                                                                        annealing_factor = annealing_factor
                                                                      )

    total_loss += elbo_Loss
    history['m1m2']['elbo'].append( elbo_Loss )
    history['m1m2']['image1_bce'].append( image1_bce )
    history['m1m2']['image1_rmse'].append( _rmse( image1.view( -1, 1 * 28 * 28 ), recon_image1.view( -1, 1 * 28 * 28 ) ) )
    history['m1m2']['image2_bce'].append( image2_bce )
    history['m1m2']['image2_rmse'].append( _rmse( image2.view( -1, 1 * 28 * 28 ), recon_image2.view( -1, 1 * 28 * 28 ) ) )
    history['m1m2']['speech_mse'].append( speech_mse )
    history['m1m2']['speech_rmse'].append( _rmse( speech, recon_speech ) )
    history['m1m2']['label_ce'].append( label_ce )
    history['m1m2']['label_acc'].append( _acc( y, recon_label ) )
    

    #  find the ELBO Loss by using speech data & one image data -> USE speech & image1
    recon_image1, _, recon_speech, recon_label, mu, logvar = model( firstImage = image1, speech = speech )

    elbo_Loss, image1_bce, image2_bce, speech_mse, label_ce = elboLoss( recon_image1 = recon_image1, image1 = image1,
                                                                        recon_speech = recon_speech, speech = speech,
                                                                        recon_label = recon_label, y = y , mu = mu,
                                                                        logvar = logvar, lambda_image = lambda_image,
                                                                        lambda_speech = lambda_speech,
                                                                        annealing_factor = annealing_factor
                                                                      )

    total_loss += elbo_Loss
    history['m1m3']['elbo'].append( elbo_Loss )
    history['m1m3']['image1_bce'].append( image1_bce )
    history['m1m3']['image1_rmse'].append( _rmse( image1.view(-1, 1 * 28 * 28), recon_image1.view(-1, 1 * 28 * 28) ) )
    history['m1m3']['image2_bce'].append( image2_bce )
    history['m1m3']['image2_rmse'].append( _rmse( image2.view(-1, 1 * 28 * 28), recon_image2.view(-1, 1 * 28 * 28) ) )
    history['m1m3']['speech_mse'].append( speech_mse )
    history['m1m3']['speech_rmse'].append( _rmse( speech, recon_speech ) )
    history['m1m3']['label_ce'].append( label_ce )
    history['m1m3']['label_acc'].append( _acc( y, recon_label ) )
    

    #  find the ELBO Loss by using speech data & one image data -> USE speech & image2
    _, recon_image2, recon_speech, recon_label, mu, logvar = model( secondImage = image2, speech = speech )

    elbo_Loss, image1_bce, image2_bce, speech_mse, label_ce = elboLoss( recon_image2 = recon_image2, image2 = image2,
                                                                        recon_speech = recon_speech, speech = speech,
                                                                        recon_label = recon_label, y = y, mu = mu,
                                                                        logvar = logvar, lambda_image = lambda_image,
                                                                        lambda_speech = lambda_speech,
                                                                        annealing_factor = annealing_factor
                                                                      )
    total_loss += elbo_Loss
    history['m2m3']['elbo'].append(elbo_Loss)
    history['m2m3']['image1_bce'].append(image1_bce)
    history['m2m3']['image1_rmse'].append(_rmse(image1.view(-1, 1 * 28 * 28), recon_image1.view(-1, 1 * 28 * 28)))
    history['m2m3']['image2_bce'].append(image2_bce)
    history['m2m3']['image2_rmse'].append(_rmse(image2.view(-1, 1 * 28 * 28), recon_image2.view(-1, 1 * 28 * 28)))
    history['m2m3']['speech_mse'].append(speech_mse)
    history['m2m3']['speech_rmse'].append(_rmse(speech, recon_speech))
    history['m2m3']['label_ce'].append(label_ce)
    history['m2m3']['label_acc'].append(_acc(y, recon_label))
    history["total_loss"].append(total_loss)

    return history, total_loss.float()
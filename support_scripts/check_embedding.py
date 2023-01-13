import numpy as np
from config import *
from paths import *

if __name__ == "main":
  
    with open( os.path.join( lookupEmbeddingDir, "lookup_mnist_language_id.npy" ), 'wb' ) as f:
        np.save( f, np.random.rand( numMNISTLang, mnistLangEmbeddingDims ) )

    with open( os.path.join( lookupEmbeddingDir, "lookup_speaker_id.npy" ), 'wb' ) as f:
        np.save( f, np.random.rand( numSpeakers, speakerEmbeddingDims ) )

    with open( os.path.join( lookupEmbeddingDir, "lookup_digit.npy" ), 'wb' ) as f:
        np.save( f, np.random.rand( numDigit, digitEmbeddingDims ) )

    with open( os.path.join( lookupEmbeddingDir, "emb_matrix_synthetic_mu.npy" ), 'wb' ) as f:
        np.save( f, np.random.rand( mnistLangEmbeddingDims + speakerEmbeddingDims + digitEmbeddingDims, 
                                    synthModalityDims
                                  )
               )

    with open( os.path.join(lookupEmbeddingDir, "emb_matrix_synthetic_sigma.npy"), 'wb') as f:
        np.save( f, np.random.rand( mnistLangEmbeddingDims + speakerEmbeddingDims + digitEmbeddingDims, 
                                    synthModalityDims
                                  )
               )
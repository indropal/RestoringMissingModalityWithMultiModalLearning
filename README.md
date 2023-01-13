<h3>Restoring Missing Modality With MultiModal Learning</h3>

<hr style="border: 4px double grey"></hr>

This project explores Multi-modal learning to obtain higher quality represenations for modailities especially for instances when they occur together at the same time.
<br>

Recent progress in <b>Generative Deep Learning</b> has given rise to state-of-the-art algorithms like Variational Autoencoder (VAE), GANs and their variations. This project makes use of a <b>Multimodal Variational Autoencoder</b> <em>(<b>MVAE</b>)</em> that uses generative processes to learn joint distributions of multi-modal input and handle instances of missing data.
<br>

The project takes inspiration from the following research article - <em><a href="https://arxiv.org/pdf/1802.05335.pdf">"Multimodal Generative Models for Scalable Weakly-Supervised Learning"</a></em>, building upon the concept of <b><em>weakly supervised learning</em></b> to support missing modal data and capture joint representations of modalities - as presented by the authors of this article.
<br>

<h4><em>NOTE: Extensions to this project is currently under development</em></h4>

<img src="https://github.com/indropal/RestoringMissingModalityWithMultiModalLearning/blob/main/DocImages/MVAE_Reconstructed%20Digit%205.png"></img>
<p><em>Sample reconstruction of Bengali Digit 5 for multi-modal speech (English) & image (MNIST Arabic) inputs</em></p>


<hr style="border: 4px double grey"></hr>
<h3>Objectives & Methodology of the project</h3>

The <em>Multimodal Variational Autoencoder</em> builds upon the <em>VAE</em> to train a generative model that learns representations bridging between <em>latent variables</em> & <em>observations</em> along with training an <em><b>inference</b></em> network from observations to latents (which is the converse association).
<br>
In order to scale & consider all possible combinations of modalities, the <b>inference network</b> is extended to a <b><a href="https://en.wikipedia.org/wiki/Product_of_experts">product-of-expert</a></b> variation by assuming conditional independence amongst modalities.
<br>

In this project, the inference network of the <em>Multimodal Variational Autoencoder</em> is built using modal data from three different modalities consisting of speech data of digit recordings & MNIST handwritten digit images in two different languages. The objective is to reconstruct one handwritten digit modality in a specific language given other two modalities as input to the MVAE network. 
<br>

The assumption is that the data for the modalitites are conditionally independent for a common learnt latent variable.

<h4>Details about the Dataset</h4>

<b>Technology Stack </b>: <em>Python, PyTorch</em>

The datasets used for this project are as follows:
<ul>
    <li>The MNIST Image Dataset is obtained from <a href="https://github.com/jwwthu/MNIST-MIX">MNIST-MIX</a> data.</li>
    <li>The Speech dataset of Digit Recordings is obtained from <a href="https://github.com/Jakobovski/free-spoken-digit-dataset">Free Spoken Digit Dataset</a> composed by 6 different speakers.
    </li>
</ul>
The Speech Data is decomposed and preprocessed by extracting <a href="https://en.wikipedia.org/wiki/Mel-frequency_cepstrum">Mel-frequency cepstral coefficients</a> and organizing them into Tensors for training the generative deep learning model.

<h4>Methodology</h4> :

The project currently involves a trained MVAE on MNIST handritten digits data for two languages 'Bangla' (Bengali) & 'Arabic' while speech data for 'English' is used as the third modality. <br>

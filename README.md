<h3>Restoring Missing Modality With MultiModal Learning</h3>

<hr style="border: 4px double grey"></hr>

This project explores Multi-modal learning to obtain higher quality represenations for modailities especially for instances when they occur together at the same time.
<br>

Recent progress in <b>Generative Deep Learning</b> has given rise to state-of-the-art algorithms like Variational Autoencoder (VAE), GANs and their variations. This project makes use of a <b>Multimodal Variational Autoencoder</b> <em>(<b>MVAE</b>)</em> that uses generative processes to learn joint distributions of multi-modal input and handle instances of missing data.
<br>

The project takes inspiration from the following research article - <em><a href="https://arxiv.org/pdf/1802.05335.pdf">"Multimodal Generative Models for Scalable Weakly-Supervised Learning"</a></em>, building upon the concept of <b><em>weakly supervised learning</em></b> to support missing modal data and capture joint representations of modalities - as presented by the authors of this article.
<br>

The <em>Multimodal Variational Autoencoder</em> builds upon the <em>VAE</em> to train a generative model that learns representations bridging between <em>latent variables</em> & <em>observations</em> along with training an <em><b>inference</b></em> network from observations to latents (which is the converse association).

<b><em>Extensions to this project is currently being explore</em></b>
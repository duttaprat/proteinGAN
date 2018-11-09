## Findings
### 2018-08-10
- One-hot-encoding produce sequences that contains a variety of amino acids (18-20) very early on. 
Generated samples also match distribution of amino acids fairly well.
(
wgan/mini_sample/sngan/protein_tcn_batch_64_lr_0.0004_b1_0.0_b2_0.999_dim_16_image_3x192)  
However, it collapses at some point and starts generating only a couple 
of amino acids. (perfect discriminator, stale generator)
- Training embeddings during GAN mini-max game is not reasonable since 
embeddings being updated by discriminator, since gives discriminator 
an advantage. Results are supporting this.
Generated proteins usually consisted on single amino acid (all values are the same).
### 2018-08-14
- Handcrafted embeddings suffers from only-one amino acid problem. 
This problem disappears if only 3 features are used from embedding 
(kernel height ==  number of embedding features). 
- Full embeddings can be used with reduced kernel size 
(3x3 works the best, bigger kernel makes it harder to train).
- Removing dilation rate did not have significant improvement. 
- Increasing size of noise also did not have any significant impact. 
- 0 (empty embedding is not being generated)
- **A model that generates samples which are similar to 3.6.7.1 class** 
(/home/donatasrep/workspace/PREnzyme/weights/protein//wgan//mini_sample/sngan/protein_tcn_batch_256_lr_0.0004_b1_0.0_b2_0.999_dim_32_image_38x128_embedding)
Increased number of parameters significantly improved performance.

### 2018-08-16
- Learning rates, loss calculations(hinge vs wasterstein) does not have significant impact on performance. 
Same true with dilation.
- Growing height of the sequences makes the model to collapse.
- Two major issues are: 
    - mode collapse after training for a while
    - discriminator becoming perfect
    
### 2018-09-01

- In this particular scenario deconvolutions work better than subpixel architecture (Requires more information to understand why).
- Nearest neighbourgh as upsampling techinique works very poorly.
- Try a U-NET architecture for ReactionGAN (cGAN). It did not perform well.

### 2018-10-01
- Using dilation rate 2 massively improves results. 
- Models suffered from nearly perfect discriminator and mode collapse.
- Adding significant amount of noise to embeddings + minibatch discriminator minigated mode collapse for ProteinGAN. 
However, in comparison to ProGAN implementation, ProteinGAN requires minibatch discriminator results are concatenated 
with ratio 1:1, meaning that rezults from  minibatch discriminator is the same shape as the layer it is concatecated 
with. That is neither expected not explained. Ideally, minibatch discriminator should not require that much weight to 
keep generator from mode collapse (needs investigation).
- A need for numeric evaluation of the results.  
## 2018-07-29
- Automation scripts 
- First version of CGAN

## 2018-08-05
- Auto encoder (with 98% accuracy)
- Improved CGAN by incorporating auto encoder
- Experimented with SNGAN
= Created a framework to store experiment results

## 2018-08-12
- Compared and tuned WGAN-GP and SNGAN
- Implemented dilations in resnet blocks of GANs
- Polish a framework to store experiment results (amino acid distribution, blast, etc.)

## 2018-08-19
- Started training GANs with proteins
- Added handcrafted protein embeddings to GANs
- Ran multiple experiments with different parameters, evaluated performance and documented results

### 2018-08-26
- Code refactoring
- Conv with stride 2 and deconv instead of avg pooling and nearest neighbour
- Prepared tfrecords
- Helped iGEM team
- Tested subpixel approach. No signifcant impact on result compared with deconv


### 2018-09-02
- Implemented U-NET for conditional GAN. No good results
- Changed padding from SAME to VALID and performed padding manually using REFLECT method.

### 2018-09-09
- Tested performance of model with extra embeddings. No significant impact.
- Helped with SignalAlign, MMSEQS

### 2018-09-16
- Implemented mini batch discriminator
- Conducted multiple tests to find solution for mode collapse
- Simplify architecture

### 2018-09-23
- N/A

### 2018-09-30 
- N/A

### 2018-10-07
- Implemented self-attention layer
- Conducted multiple experiments to investigate mode collapse further.
- Improved architecture. First time got BLAST results what had e-50

### 2018-10-14
- Reviewed and improved embeddings
- Added noise to emdebbings
- Added back mini batch discriminator. This time it worked
- Implemented label swap - make results poor.
- Documentation for iGEM.

### 2018-10-21
- Finished documenation for iGEM
- Trained ProteinGAN and ReactionGAN for long time.
- Analyzed results.

### 2018-10-28
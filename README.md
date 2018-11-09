# PREnzyme
Developed two different generative network architectures that may be used to produce de-novo protein sequences 
(**ProteinGAN**) as well as to generate protein sequences given a specific catalytic reaction (**ReactionGAN**).


## Motivation

One of the most challenging task in synthetic biology is to create novel synthetic parts, in particular synthetic 
catalytic biomolecules.

Catalytic biomolecules are a cleaner and greener substitute to chemical catalyzers used all over the world. 
The catalytic biomolecules offer an enormous 17 orders of magnitude chemical reaction acceleration as well as excellent 
stereo-, chemo- and regio-selectivity in aqueous environments. Yet, one of the main drawbacks of using enzymes is that 
for many important chemical reactions, efficient enzymes have not yet been discovered or engineered.

That said, identifying enzyme amino acid sequence with the required novel or optimized reaction is a challenging task 
because the sequence space is incomprehensibly large.

### ProteinGAN

ProteinGAN - generative adversarial network architecture that we developed and optimized to generate de novo proteins. 
Once trained, it outputs a desirable amount of proteins with the confidence level of sequences belonging to a functional 
class it was trained for.


### ReactionGAN

ReactionGAN is a conditional generative adversarial network architecture that is capable of generating de novo proteins 
for a given reaction. Once trained, you can use it by simply providing a list of reactions. The output will contain 
corresponding generated sequences.
The goal of this new network is to not only learn the rules of protein sequences, but also to learn the connection between
enzyme sequences and reactions they catalyze

## Useful links

- Database of annotated enzymes by its function - http://www.uniprot.org/. Better to use manually annotated sequences. 
EC number describes enzyme class/subclasses (ex.: 1.1.20.2).
- Database of enzyme reactions: https://www.expasy.org/.
- Heidelber project of similar subject - http://2017.igem.org/Team:Heidelberg/Software/DeeProtein
- Paper on generating sequences (DNA) using GANs - https://arxiv.org/pdf/1712.06148.pdf

Also good reads:
- https://arxiv.org/pdf/1703.10663.pdf
- https://arxiv.org/ftp/arxiv/papers/1701/1701.08318.pdf
- https://arxiv.org/pdf/1803.01271.pdf
- https://arxiv.org/pdf/1809.11096.pdf
- https://arxiv.org/pdf/1710.10196.pdf
- https://arxiv.org/abs/1802.05957.pdf
- https://arxiv.org/pdf/1606.03498.pdf
- https://arxiv.org/pdf/1802.05957.pdf
- https://arxiv.org/pdf/1511.07122.pdf
- https://arxiv.org/pdf/1805.08318.pdf
- https://arxiv.org/pdf/1802.05637.pdf

More info: http://2018.igem.org/Team:Vilnius-Lithuania-OG/Software
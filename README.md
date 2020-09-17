# A Variational Autoencoders (VAE) toolkit to study proteins oligomerization

In this code, I implemented a python module capable of constructing
VAE models, in order to study protein-protein interactions and
oligomerization. The code inputs a multidimensional matrix,
containing the pairwise distances between amino acids of different
monomers, which need to be calculated by users from a molecular
dynamics (MD) simulation. The implemented machine learning module
then uses this input matrix to construct a VAE model to map the data
into a latent space. Different hyperparameters of the module
including, the deep neural network layers and nodes in the VAE
model, learning rate, number of epochs, and the dimension of the
latent space can be optimized, using Hyperopt libraries in python.
The VAE model itself takes advantage of different Pytorch libraries,
including: nn and autograd.
The latent space constructed here can contain several clusters,
representing different protein oligomerization states, as well as their
differences and similarities with amino acid resolution.
It is important to mention that the oligomerization states each
should be sampled in the MD simulation, otherwise, the implemented
VAE algorithm cannot predict them. We have tested this module on a
40 microsecond KRAS dimer trajectory (simulated on Anton2 machine),
and the constructed latent space was able to show all the
dimerization states of KRAS monomers and their similarities and
differences (unpublished - in preparation). 


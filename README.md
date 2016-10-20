# JEDI
Joint Estimation of Dictionary and Image (from compressive samples)

This is the source code for my paper that is under revision in IEEE-Transactions on Computaational Imaging. A draft of the paper is placed in `/doc/`.

#### Installation

I use [SPAMS](https://github.com/samuelstjean/spams-python) for LASSO optimization. SPAMS library is located in the `/lib/` directory. Please follow the SPAMS installation instructions for Linux and OSX. For Windows, it is easier to use SPAMS's binaries that can be downloaded from http://spams-devel.gforge.inria.fr/downloads.html

#### Running

 - For compressed sensing using the JEDI algorithm run `python JEDI_compressed_sensing.py`.
 - For image inpainting using the JEDI algorithm run `python JEDI_inpainting.py`.
 
 

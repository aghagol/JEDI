# JEDI
Joint Estimation of Dictionary and Image (from compressive samples)

This is the source code for my paper that is under revision in IEEE-Transactions on Computational Imaging. A draft of the paper is placed in `/doc/`.

#### Installation

First, do `git clone --recursive https://github.com/aghagol/JEDI.git` where you want the local repo to be placed.

I use [SPAMS](https://github.com/samuelstjean/spams-python) for LASSO optimization. SPAMS library is located in the `/lib/` directory. Please follow the SPAMS installation instructions for Linux and MacOS. For Windows, it is easier to use SPAMS's binaries that can be downloaded from http://spams-devel.gforge.inria.fr/downloads.html

I have had problems with installing SPAMS on MacOS. Here is how I could successfully install SPAMS and link it to my Python installation, e.g. `/usr/local/bin/python`:

 - Clear `cc_flags` and `link_flags` (make them empty lists) in `spams/setup.py` under MacOS.
 - `python setup.py build`
 - `sudo python setup.py install --prefix=/usr/local/`

#### Running

 - For compressed sensing using the JEDI algorithm run `python JEDI_compressed_sensing.py`.
 - For image inpainting using the JEDI algorithm run `python JEDI_inpainting.py`.
 
 

# HipoMap

# Setup
1. Download ALL the files and folders.
2. Setup Anaconda (Linux) if you havent already
3. Create the environment (conda env create -f linux_model.yaml)
4. If you have anaconda3 already installed at your home directory, you can skip this step. Otherwise, edit linux_tcgajob.sh and change "source" to where anaconda3 is.

# Running it
1. Create a folder in the HipoMap directory named "Slides" with a number at the end, no space. Ex: "Slides1" or "Slides2"
2. Then run this command where "n" would be the number that you appened to the Slides folder (bash linux_tcgajob.sh n).
   Ex: You have "Slides1" then run "bash linux_tcgajob.sh 1"

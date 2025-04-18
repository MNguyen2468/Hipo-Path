#!/bin/bash
#SBATCH --job-name=linux_tcga
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --nodes=1             # Number of nodes
#SBATCH --ntasks=1            # Total number of tasks

#The file name and output
TODAY_DATE=$(date +'%m%d%Y')
LOG_FILE="outputmn${TODAY_DATE}-$1.log"
exec > "$LOG_FILE"
# Load necessary modules (e.g., Python)
source $HOME/anaconda3/bin/activate
conda activate linux_model
#Run the Python script with datasets
python3 Hipomap.py "$1"

#bash linux_tcgajob.sh 11
#find ./Unprocessed\ slides -type f -name "*.svs" -exec mv {} ./Slides \;

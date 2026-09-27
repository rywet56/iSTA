# create conda environmnet with .yaml file
conda env create -f environment.yml
# activate it
conda activate anno_2d
# install the kernel so it can be used within jupyter lab notebook.
/opt/anaconda3/envs/anno_2d/bin/python -m ipykernel install --user --name anno_2 --display-name "Python (anno_2d)"

# run the iSTA.py file to start the annotation process.
/opt/anaconda3/envs/anno_2d/bin/python iSTA.py



rsync -avu --progress -e "ssh -i /Users/manuelneumann/.ssh/id_rsa -p 30480" ubuntu@134.176.27.78:/mnt/storage/data/spatial_transcriptomics_freddy/AT_06_fc_018/projects/arabidopsis/processed_data/at_06_rep_1_section_9/multimodal/stitched_spots.h5ad ./


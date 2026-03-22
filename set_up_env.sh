# create conda environmnet with .yaml file
conda env create -f environment.yml
# activate it
conda activate anno_2d
# in stall the kernel so it can be used within jupyter lab notebook.
/opt/anaconda3/envs/anno_2d/bin/python -m ipykernel install --user --name anno_2 --display-name "Python (anno_2d)"

# run the iSTA.py file to start the annotation process.
/opt/anaconda3/envs/anno_2d/bin/python iSTA.py
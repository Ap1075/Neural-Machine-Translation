# Steps to run the translation model:

1. Clone this repository in local  a directory using: git clone <dir>
2. Execute dev_setup.ssh to create required directories (temp,bin,corpora and output): bash dev_setup.sh
3. Copy the contents of corpora folder in the repo into the corpora folder in your local directory.
4. Go to mactrans/bin and run train_mt.py using appropriate flag values according to dataset.
5. After training, the model is saved in an hdf5 file. A pickle file with the same prefix as the hdf5 model file stores the fitted
   tokenizers along with a dictionary containing some essential variables and the path to the trained model, later used for translation.
5. To translate sentences, the sentence to be translated (source) is passed as a command line argument along with the same prefix passed 
   during training for the model and pickle files.

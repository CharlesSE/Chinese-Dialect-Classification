# Chinese-Dialect-Classification
Project for TDT4310: Classification of Mandarin text from Mainland China and Taiwan.

To run the POS tagger you need to download the Stanford Tagger (Full Version): https://nlp.stanford.edu/software/tagger.shtml#Download

The Stanford tagger should be placed in the same folder as the rest of the code, so that its path matches the path_to_model and path_to_jar variables. Similarly, the corpus should also be placed in this folder, so that its path matches TRAIN_PATH and TEST_PATH.

For the deep learning models, the files for the models and weights are provided. Using those you don't have to train the models again. 

The ML classification models are located in the file Classifier.py. 
The DL models are in the files CNN.py, DNN.py, LSTM.py, and BILSTM.py.

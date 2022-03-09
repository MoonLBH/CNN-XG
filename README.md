# CNN-XG  
## Overview
CNN-XG is a deep learning-based model for CRISPR/Cas9 sgRNA on-target cleavage efficacy prediction. It is composed of two major components: a CNN as the front-end for extracting gRNA and epigenetic features as well as an XGBoost as the back-end for regression and predicting sgRNA cleavage efficiency. 

## Requirement:  
* **Python packages:**   
 * [numpy](https://numpy.org/) 1.19.2 
 * [pandas](https://pandas.pydata.org/) 1.2.4 
 * [scikit-learn](https://scikit-learn.org/stable/) 0.23.2
 * [scipy](https://www.scipy.org/) 1.6.2  
 * [Keras](https://keras.io/) 2.4.3    
 * [Tensorflow](https://tensorflow.google.cn/) 2.4.1    
   
  

## Content
* **./data:** the training and testing examples with sgRNA sequence and corresponding epigenetic features and label indicating the on-target cleavage efficacy 
* **./new_data:** three cell line datasets for generalization capability testing and five SpCas9 variant datasets
* **./weights/weights.h5:** the well-trained weights for our model in four initial datasets
* **./CNN-XG.py:** the python code, it can be ran to reproduce our results  

## Usage
#### **python CNN-XG.py**       
**Note:**  
* The input training and testing files should include gRNA sequence with length of 23 bp and four "A-N" symbolic corresponding epigenetic features seuqnces with length of 23 as well as label in each gRNA sequence.    
* The train.csv, test.csv can be replaced or modified to include gRNA sequence and four epigenetic features of interest  

## Demo instructions  
#### **Input (gRNA sequence and four epigenetic features):**               
* #### **Data format:**      
*   **sgRNA sequence:** TGAGAAGTCTATGAGCTTCAAGG (23bp)      
*   **ctcf:** NNNNNNNNNNNNNNNNNNNNNNN (length=23)      
*   **dnase:** AAAAAAAAAAAAAAAAAAAAAAA (length=23)      
*   **h3k4me3:** NNNNNNNNNNNNNNNNNNNNNNN (length=23)      
*   **rrbs:** NNNNNNNNNNNNNNNNNNNNNNN (length=23)    
#### **Load weights (Pre-trained weight file):**        
./weights/weights.h5   
#### **Run script:**       
python ./CNN-XG.py   
#### **Output (Predicted activity score for gRNA):** 


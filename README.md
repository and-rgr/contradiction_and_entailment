# Contradictory My Dear Watson 

### Introduction

We train a neural network on natural language inference. We use the dataset provided by the Kaggle competition [Contradicty My Dear Watson](https://www.kaggle.com/competitions/contradictory-my-dear-watson). This dataset contains premise-hypothesis pairs in fifteen different languages, including: Arabic, Bulgarian, Chinese, German, Greek, English, Spanish, French, Hindi, Russian, Swahili, Thai, Turkish, Urdu, and Vietnamese. We classify these pairs as **entailment**, **contradiction**, or **neutral**.

The code is based on the following notebook: [Detecting Contradictions and Entailment in Multilingual Text](https://github.com/sukanyabag/Detecting-Contradictions-and-Entailment-in-Multilingual-Text/tree/main/Detecting%20Contradictions%20in%20Multilingual%20Text)

### Model Used
We use the pretrained model **joeddav/xlm-roberta-large-xnli**, which is based on RoBERTa, itself an enhancement of the BERT transformer. We choose this model because it is fine tuned for natural language inference in multiple languages, which idealy suits this task.

### Dependencies 
- transformers, version 4.19.2
- sentencepiece, version 0.1.96

### Libraries Used
- datetime
- pandas
- tensorflow
- transformers
- sentencepiece

### Data Source
Data is pulled from the **dataset.csv** file, and separated into training and validation sets during training.

### Performance
We achieve an accuracy of 0.9241 on the validation set, and a loss of 0.4533.

### Output
After training, we save the model itself as an **.hdf5** file,, the training history as a **.csv** file, and the hyperparameters used as a **.txt** file.

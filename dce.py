import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# requires the sentencepiece package (unsupervised text tokenizer and detokenizer)
from transformers import TFAutoModel,AutoTokenizer

# used for the BertTokenizer
# from transformers import BertTokenizer, TFBertModel

## see the entire dataframe width in the console
pd.set_option('display.expand_frame_repr', False)

## to silence warning - what?
os.environ["WANDB_API_KEY"] = "0"

# check the TPU
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy() # for CPU and single GPU

print('Number of replicas:', strategy.num_replicas_in_sync)

# see available files
for dirname, _, filenames in os.walk('./data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# load data
train=pd.read_csv('./data/contradictory-my-dear-watson/train.csv')
test=pd.read_csv('./data/contradictory-my-dear-watson/test.csv')

# inspect data
print(train.shape)
print(test.shape)

print(train.tail(10))
print(test.tail(10))

print(train.premise.values[1], '\n', train.hypothesis.values[1], '\n', train.label.values[1])

labels, frequencies = np.unique(train.language.values, return_counts = True)

plt.figure(figsize = (10,10))
plt.pie(frequencies,labels = labels, autopct = '%1.1f%%')
plt.show()

print(train.label.value_counts())



### NEW STUFF

# we download a pre-trained tokenizer
# RoBERTa: A Robustly Optimized BERT Pretraining Approach
tokenizer=AutoTokenizer.from_pretrained('joeddav/xlm-roberta-large-xnli')

# might also use
# tokenizer2 = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# batch_encode_plus() generates a dictionary with the input_ids, token_type_ids and the attention_mask as list for each input sentence
# sentences with different lengths need to be truncated or padded, before combined into a single tensor
# the attention mask is a binary tensor indicating the position of the padded indices so that the model does not attend to them
# documentation:
#     https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.tokenization_utils_base.PreTrainedTokenizerBase.batch_encode_plus
train_enc=tokenizer.batch_encode_plus(train[['premise','hypothesis']].values.tolist(),padding='max_length',max_length=100,truncation=True,return_attention_mask=True)
test_enc=tokenizer.batch_encode_plus(test[['premise','hypothesis']].values.tolist(),padding='max_length',max_length=100,truncation=True,return_attention_mask=True)

# combine tokenization objects into training and test tensors
# first time error:
#     Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU.
#     CUDA toolkit installation may require Visual Studio as well
#     added to path
#       C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\libnvvp;
#       C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\bin;
#     need to install cuDNN SDK 8.1.0, but it requires NVIDIA Developer Program Membership
train_tf1=tf.convert_to_tensor(train_enc['input_ids'],dtype=tf.int32)
train_tf2=tf.convert_to_tensor(train_enc['attention_mask'],dtype=tf.int32)
train_input={'input_word_ids':train_tf1,'input_mask':train_tf2}

test_tf1=tf.convert_to_tensor(test_enc['input_ids'],dtype=tf.int32)
test_tf2=tf.convert_to_tensor(test_enc['attention_mask'],dtype=tf.int32)
test_input={'input_word_ids':test_tf1,'input_mask':test_tf2}


print(train_enc[100])


# QUESTION - what does strategy.scope() do?
with strategy.scope():
    # instantiates a Keras tensor
    input_ids = tf.keras.Input(shape = (100,), dtype = tf.int32, name = 'input_word_ids')
    input_mask = tf.keras.Input(shape = (100,), dtype = tf.int32, name = 'input_mask')

    # documentation: https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#tfautomodel
    roberta = TFAutoModel.from_pretrained('joeddav/xlm-roberta-large-xnli')
    # at this point roberta([input_ids,input_mask]) is a tensorflow object, that contains two keras tensors
    # QUESTION - what does this do??
    roberta = roberta([input_ids,input_mask])[0]

    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling1D
    # "Global average pooling operation for temporal data"
    # https://stackoverflow.com/questions/54493738/keras-difference-between-averagepooling1d-layer-and-globalaveragepooling1d-laye
    # QUESTION - is average pooling similar to max pooling for convolution networks?
    output = tf.keras.layers.GlobalAveragePooling1D()(roberta)
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
    # "regular densely-connected NN layer"
    output = tf.keras.layers.Dense(3, activation = 'softmax')(output)

    model = tf.keras.Model(inputs = [input_ids,input_mask], outputs = output)
    model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-5),
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])
    model.summary()


early_stop = tf.keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)

# train the model?
# uses up space on D: for some reason
model.fit(train_input,train.label,validation_split = 0.2,epochs=5,batch_size=16*strategy.num_replicas_in_sync,callbacks=[early_stop],verbose=1)

    # 2022-01-12 01:34:51.737439: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 1024008192 exceeds 10% of free system memory.
    # WARNING:tensorflow:Gradients do not exist for variables ['tfxlm_roberta_model/roberta/pooler/dense/kernel:0', 'tfxlm_roberta_model/roberta/pooler/dense/bias:0'] when minimizing the loss.


pred=[np.argmax(i) for i in model.predict(test_input)]
print(pd.DataFrame(pred).value_counts())


pd.DataFrame({'id':test.id,
              'prediction':pred}).to_csv('submission.csv',index=False)
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'AI Learning'))
	print(os.getcwd())
except:
	pass

#%%
import pickle
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import keras
from sklearn.model_selection import train_test_split
import keras.backend as K
import seaborn as sn
import pandas as pd
from sklearn.metrics import *
import sklearn
import re
np.random.seed(10)


#%%
def predictBinary(x):
    g = list()
    for i in x:
        if i >= 0.5:
            g.append(1.)
        else:
            g.append(0.)
    return g

def predictBinaryAdv(x):
    g = list()
    s = sum(x)/float(len(x))
    for i in x:
        if i >= s:
            g.append(1.)
        else:
            g.append(0.)
    return g

def predictSoftmax(x):
    return [np.argmax(i) for i in x]


#%%
# Load the Sarcasm Analysis Data
data = pickle.loads(open('./Processed/nnlm128_Sarcasm.txt', 'rb').read())
X = np.array(data[0])
labels = list()
for i in data[1]:
    if i == 0:
        labels.append([1])
    else:
        labels.append([0])
        
Y = np.array([[float(i[0])] for i in labels])#data[1]])
#     labels])#


#%%

Y[1]
Y = keras.utils.to_categorical(Y)
Y[0]


#%%
train_x, test_x, train_y, test_y = train_test_split(X[:3000000],Y[:3000000],test_size=0.2)


#%%
len(Y)


#%%
# Sarcasm Model -->
inputs = tf.keras.Input(shape=X[0].shape)
x = layers.Dense(70, kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
x = layers.ReLU()(x)
#x = layers.GaussianDropout(0.3)(x)
#x = layers.BatchNormalization(axis=1)(x)
#x = layers.Dropout(0.2)(x)
#x = layers.BatchNormalization()(x)
#x = layers.Dense(64, activation = "selu",kernel_regularizer=keras.regularizers.l2(0.001))(x)
#x = layers.BatchNormalization(axis=1)(x)
#x = layers.Dropout(0.5)(x)
x = layers.Dense(10, activation = "sigmoid", kernel_regularizer=keras.regularizers.l2(0.001))(x)
stars = layers.Dense(2, activation='softmax', name="stars")(x)#(len(uniqs), activation='softmax')(x) 

sarcasmmodel = tf.keras.Model(inputs = inputs, outputs = stars)
sarcasmmodel.compile(optimizer = 'rmsprop', loss='categorical_crossentropy', metrics=['accuracy', 'mae'])
sarcasmmodel.summary()


#%%
# simple early stopping
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, restore_best_weights=True)
history = sarcasmmodel.fit(train_x[:], train_y[:], validation_data=(test_x, test_y), epochs=100, shuffle=True, callbacks=[es])
sarcasmmodel.evaluate(test_x, test_y)


#%%
history_dict = history.history
history_dict.keys()
predict_y = sarcasmmodel.predict(test_x)


#%%
import matplotlib.pyplot as plt

acc = history_dict['accuracy']
loss = history_dict['loss']
mae = history_dict['mae']
val_acc = history_dict['val_accuracy']
val_loss = history_dict['val_loss']
val_mae = history_dict['val_mae']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, mae, 'b', label='MAE')
plt.plot(epochs, val_loss, 'ro', label='Validation loss')
plt.plot(epochs, val_mae, 'r', label='Validation MAE')
#plt.title('Training and validation loss')
plt.xlabel('Epochs')
#plt.ylabel('Loss')
plt.legend()

plt.show()
plt.plot(epochs, acc, 'b', label="Training Accuracy")
plt.plot(epochs, val_acc, 'r', label="ValidationAccuracy")
plt.legend()
plt.show()

cm = confusion_matrix(predictSoftmax(test_y), predictSoftmax(predict_y))
print(cm)
cm = [i/sum(i) for i in cm]
df_cm = pd.DataFrame(cm, range(2),
                  range(2))
#plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 14})# font size


#%%
accuracy_score(predictSoftmax(Y), predictSoftmax(model.predict(X)))*100


#%%
# Save Sarcasm Model -->
model.save('nnlm_sarcasm.h5')


#%%
model = tf.keras.models.load_model('nnlm_sarcasm.h5')
sarcasmmodel = model


#%%


cm = confusion_matrix(predictSoftmax(Y), predictSoftmax(model.predict(X)))
print(cm)
cm = [i/sum(i) for i in cm]
df_cm = pd.DataFrame(cm, range(2),
                  range(2))
#plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 14})# font size


#%%
# Load the Sentiment Analysis Data
data = pickle.loads(open('./Processed/nnlm128_Sent.txt', 'rb').read())
X = data[0]
#X = np.array(data[0])
labels = list()
#for i in data[1]:
#    if i == 0:
#        labels.append([1])
#    else:
#        labels.append([0])
        
Y = np.array([[float(i)] for i in data[1]])
#     labels])#
Y.shape
Y[0]


#%%
X = np.array(X)
X.shape


#%%
# Freeze the sarcasm model layers -->
for i in sarcasmmodel.layers:
    i.trainable = False

for i in sarcasmmodel.layers:
    print(i.trainable)


#%%

Y = keras.utils.to_categorical(Y)
train_x, test_x, train_y, test_y = train_test_split(X[:300000],Y[:300000],test_size=0.2)


#%%
# Sentiment Model -->
#inputs = tf.keras.Input(shape=X[0].shape)
#x = layers.Dense(300, activation = "relu", kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
#x = layers.BatchNormalization(axis=1)(x)
x = layers.Dense(300, activation = "relu", kernel_regularizer=keras.regularizers.l2(0.001))(sarcasmmodel.input)
#x = layers.Dropout(0.5)(x)
#x = layers.BatchNormalization()(x)
x = layers.Dense(100,kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.PReLU()(x)
#g = layers.Dense(300, activation = "softmax")(x)
#x = layers.Concatenate(axis=0)([x, g])

#x = layers.BatchNormalization(axis=1)(x)
#x = layers.Dropout(0.5)(x)
x = layers.Dense(50, activation = "sigmoid", kernel_regularizer=keras.regularizers.l2(0.001))(x)
#x = layers.Dense(2, activation = "softmax", kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.Concatenate()([x, sarcasmmodel.layers[-2].output])
#x = layers.Dense(2, activation='softmax')(x)#(len(uniqs), activation='softmax')(x) 
output = layers.Dense(2, activation='softmax', name="output")(x)

sentimodel = tf.keras.Model(inputs = [sarcasmmodel.input], outputs = output)
sentimodel.compile(optimizer = 'rmsprop', loss='categorical_crossentropy', metrics=['accuracy', 'mae'])
sentimodel.summary()


#%%
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, restore_best_weights=True)
history = sentimodel.fit(train_x[:], train_y[:], validation_data=(test_x, test_y), epochs=100, shuffle=True, callbacks=[es])
sentimodel.evaluate(test_x, test_y)


#%%
history_dict = history.history
history_dict.keys()
predict_y = sentimodel.predict(test_x)


#%%
import matplotlib.pyplot as plt

acc = history_dict['accuracy']
loss = history_dict['loss']
mae = history_dict['mae']
val_acc = history_dict['val_accuracy']
val_loss = history_dict['val_loss']
val_mae = history_dict['val_mae']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, mae, 'b', label='MAE')
plt.plot(epochs, val_loss, 'ro', label='Validation loss')
plt.plot(epochs, val_mae, 'r', label='Validation MAE')
#plt.title('Training and validation loss')
plt.xlabel('Epochs')
#plt.ylabel('Loss')
plt.legend()

plt.show()
plt.plot(epochs, acc, 'b', label="Training Accuracy")
plt.plot(epochs, val_acc, 'r', label="ValidationAccuracy")
plt.legend()
plt.show()

cm = confusion_matrix(predictSoftmax(test_y), predictSoftmax(predict_y))
print(cm)
cm = [i/sum(i) for i in cm]
df_cm = pd.DataFrame(cm, range(2),
                  range(2))
#plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 14})# font size


#%%
sklearn.metrics.accuracy_score(predictSoftmax(test_y), predictSoftmax(predict_y))*100


#%%
sentimodel.save('nnlm_freshsentimodel.h5')


#%%
model = tf.keras.models.load_model('nnlm_evalModel.h5')#('nnlm_freshsentimodel.h5')
sentimodel = model


#%%
Y[0]


#%%
cm = confusion_matrix(predictSoftmax(Y), predictSoftmax(sentimodel.predict(X)))
print(cm)
cm = [i/sum(i) for i in cm]
df_cm = pd.DataFrame(cm, range(2),
                  range(2))
#plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 14})# font size


#%%
import tensorflow_hub as hub
embed_method = 'nnlm128'
embed = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1")


#%%
# Filtering and Tokening Data text
def tokenizer(data):
    badWords = ['and', 'a', 'travelling', 'to', 'of', 'Platiunum', 'techkie', 'cPs']
    #inputSet = [[x for x in i['reviewText'].replace(',',' ').split(' ')] for i in dataset[:1]]
    inputSet = [re.sub("#[A-Za-z0-9]+|@[A-Za-z0-9]+|\\w+(?:\\.\\w+)*/\\S+", "", i) for i in data]
    inputSet = [[(x) for x in re.split('\W+', re.sub("[!@#$+%*:()'-]", ' ', i)) if x != ''] for i in inputSet]
    for i in inputSet:
        for j in range(0,len(i)):
            if re.match('^\d{4}$', i[j]):
                #print(i[j])
                i[j] = 'year'
            elif re.search('[0-9]', i[j]):
                if re.search('[a-zA-Z]', i[j]):
                    i[j] = 'item'
                else:
                    i[j] = 'num'
            elif i[j] in badWords:
                i[j] = 'AXTREMOVE'
    inputSet = [re.sub('AXTREMOVE', '', '-'.join(i)).split('-') for i in inputSet]
    return inputSet

sentiDict = {0:'negative', 1:'positive'}
sarcasmDict = {1:'negative', 0:'positive'}


#%%
val = evalModel.predict(embed(["This is great"])[:1])
print(val)
sentiDict[np.argmax(val)]


#%%
X.shape


#%%
#val = sarcasmmodel.predict(embed(["this is just great"])[:1])
#print(val)
#sarcasmDict[np.argmax(val)]
evalModel = sentimodel


#%%
evalModel = tf.keras.Model(inputs = sentimodel.input, outputs = sentimodel.output)
evalModel.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy', 'mae'], )
evalModel.summary()


#%%
x = np.array(embed(["This is great product"][:1]))
x.shape
y = np.array([[0., 1.]])
y.shape


#%%



#%%
st = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.1, mode='max', verbose=1, patience=100, restore_best_weights=True)
evalModel.fit(x, y, epochs=1000, callbacks=[st])


#%%
evalModel.predict(np.array(embed(["This is good but that product is better"][:1])))


#%%
# Custom Dataset -->
custom = open("./custom_reviews", 'r').read().split('\n')
cusData = [i.split(":") for i in custom if i != '']
cusX = [i[0] for i in cusData]
cusY = [i[1] for i in cusData]
cusY = keras.utils.to_categorical(cusY)
cusX = np.array(embed(cusX))


#%%
st = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.1, mode='max', verbose=1, patience=100, restore_best_weights=True)
evalModel.fit(cusX, cusY, epochs=1000, callbacks=[st])


#%%
evalModel.save("nnlm_evalModel.h5")


#%%




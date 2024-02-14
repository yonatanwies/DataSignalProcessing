import librosa
from librosa import display
import os
import IPython.display as ipd
from IPython.display import Audio
import matplotlib.pyplot as plt

import time
import os
import numpy as np

path=os.getcwd()
lst = []
start_time = time.time()

for subdir, dirs, files in os.walk(path):
    for file in files:
        try:

            X, sr = librosa.load(os.path.join(subdir, file), res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)

            file = int(file[7:8]) - 1
            arr = mfccs, file
            lst.append(arr)

        except ValueError:
            continue

print("Time taken : %s minutes " % ((time.time() - start_time) / 60))

#This zip() function makes list of all first elements and store in X
#also it stores all the second elements in y

mfcc, emotions = zip(*lst)
print(emotions[1800])
mfcc = np.asarray(mfcc)
emotions = np.asarray(emotions)
from sklearn.model_selection import train_test_split

mfcc_train, mfcc_test, emotions_train, emotions_test = train_test_split(mfcc,emotions, test_size=0.20, random_state=42)
import numpy as np

mfcc_train = np.expand_dims(mfcc_train, axis=2)
mfcc_test = np.expand_dims(mfcc_test, axis=2)


## DEFINE MODEL FOR TRAINING
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers

model = Sequential()

model.add(Conv1D(64, 5,padding='same',input_shape=(40,1)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(4)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(4)))
model.add(Conv1D(256, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(8))
model.add(Activation('softmax'))
opt = tf.keras.optimizers.RMSprop(lr=0.00005, rho=0.9, epsilon=1e-07, decay=0.0)

model = Sequential()

model.add(Conv1D(64, 5,padding='same',input_shape=(40,1)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(4)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(4)))
model.add(Conv1D(256, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(8))
model.add(Activation('softmax'))
opt = tf.keras.optimizers.RMSprop(lr=0.00005, rho=0.9, epsilon=1e-07, decay=0.0)
model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

cnnhistory=model.fit(mfcc_train, emotions_train, batch_size=16, epochs=200, validation_data=(mfcc_test, emotions_test))


### DISPLAYING MODEL ACCURACY AND PLOTS
plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('..//Pictures//Loss_Model.png')
plt.show()

plt.plot(cnnhistory.history['accuracy'])
plt.plot(cnnhistory.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('..//Pictures//Accuracy_Model.png')
plt.show()

#loss and accuracy in testing dataset
loss,acc = model.evaluate(mfcc_test, emotions_test)
print("The accuracy of trained model is :{:5.2f}%".format(100*acc))

# predictions = model.predict_classes(mfcc_test,axis=1)
predictions = np.argmax(model.predict(mfcc_test), axis=1)
#print(predictions)

### CONFUSION MATRIX
# predictions = model.predict_classes(mfcc_test,axis=1)
predictions = np.argmax(model.predict(mfcc_test), axis=1)
#print(predictions)

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

new_Ytest = emotions_test.astype(int)
matrix = confusion_matrix(new_Ytest, predictions)
# print (matrix)
# 0 = neutral, 1 = calm, 2 = happy, 3 = sad, 4 = angry, 5 = fearful, 6 = disgust, 7 = surprised

plt.figure(figsize = (12, 10))
matrix = pd.DataFrame(matrix , index = ["Neutral","Calm","Happy","Sad","Angry","Fear","Disgust","Surprise"] , columns = ["Neutral","Calm","Happy","Sad","Angry","Fear","Disgust","Surprise"])
ax = sns.heatmap(matrix, linecolor='white', cmap='Purples', linewidth=1, annot=True, fmt='')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.savefig('..//Pictures//Confusion_matrix.png')
plt.show()


### Printing Precision, Recall, F1-Score and Support values of model
from sklearn.metrics import classification_report

report = classification_report(new_Ytest, predictions)
print(report)
#Saving our trained model as CNN_Model.h5

model.save('CNN_Model.h5')
print("MODEL SAVED")
#Loading back the saved model

our_model=keras.models.load_model('CNN_Model.h5')

loss, acc = our_model.evaluate(mfcc_test, emotions_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

### Using model for live prediction
import keras
import numpy as np
import librosa
import soundfile as sf
class modelPredictions:

    def __init__(self, path, file):
        self.path = path
        self.file = file

    def load_model(self):
        self.loaded_model = keras.models.load_model(self.path)
        #return self.loaded_model.summary()

    def predictEmotion(self):
        data, sr = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=1)
        x = np.expand_dims(x, axis=0)
#         predictedEmotion = self.loaded_model.predict_classes(x)
        predictedEmotion = np.argmax(self.loaded_model.predict(x), axis=1)
        print("The predicted emotion is : ", " ", self.convertclasstoemotion(predictedEmotion))

    @staticmethod
    def convertclasstoemotion(p):
        #predictions(int) to understandable emotion labeling
        label_conversion = {'0': 'neutral','1': 'calm','2': 'happy','3': 'sad','4': 'angry','5': 'fearful','6': 'disgust','7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == p:
                label = value
        return label
#Created the object p of class modelPredictions
p = modelPredictions(path='CNN_Model.h5',file='panic.wav')
p.load_model()
#called predictEmotion function to predict emotion type of input file
p.predictEmotion()

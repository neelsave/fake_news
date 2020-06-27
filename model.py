import pandas as pd
df=pd.read_csv('train.csv')
print(df.head())
df=df.dropna()
x=df.drop('label',axis=1)
y=df['label']

import pickle
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

voc_size=5000
messages=x.copy()
messages.reset_index(inplace=True)

import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')

#Dataset Preprocessing
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
  review = re.sub('[^a-zA-Z]', ' ',messages['title'][i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
  review = ' '.join(review)
  corpus.append(review)
  
onehot_repr = [one_hot(words,voc_size)for words in corpus]

sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)



#creating model
embedding_vector_freatures=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_freatures,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

import numpy as np
x_final=np.array(embedded_docs)
y_final=np.array(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.33,random_state=42)

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=64)

pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
final_model = pickle.load(open('model.pkl','rb'))

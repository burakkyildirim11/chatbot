import random
import json
import pickle
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('wordnet')



from nltk import pos_tag
from nltk.stem import WordNetLemmatizer


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

#Python NLTK'da Stemming ve Lemmatization, Doğal Dil İşleme için metin normalleştirme teknikleridir.
#NLTK'da Lemmatization, anlamı ve bağlamına bağlı olarak bir kelimenin lemmasını bulmanın algoritmik sürecidir.
#Lemmatizasyon genellikle, çekim sonlarını kaldırmayı amaçlayan kelimelerin morfolojik analizini ifade eder.
#Lemma olarak bilinen bir kelimenin taban veya sözlük biçimini döndürmeye yardımcı olur.



intents = json.loads(open('intents.json').read())

words = []
classses = []
documents = []
ignore_letters = ['?','!','.',',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) #pattern icindeki kelimeleri word_list'e atti
        words.extend(word_list) #word_list'i, words dizisine koydu
        documents.append((word_list, intent['tag'])) #document'e intent['tag']'i ekle
        if intent['tag'] not in classses: #classes icinde yoksa o tag
            classses.append(intent['tag']) #ekle

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters] #words'un icinden word'leri al, eger word ignore_letters'te yoksa
#ve ona lemmatizer.lemmatize(word) islemini yap; words'e ata.
#peki burada lemmatizer.lemmatize(word) ne anlama geliyor -> aldıgı kelimeyi sadelestiriyor

#List omprehensions Hk.
#List omprehensions, bir ifade(burada-> lemmatizer.lemmatize(word))  ve ardından bir for yan tumcesi, ardından sifir
#veya daha fazla for veya if yan tümceleri içeren parantezlerden olusur. Sonuc, ifadenin onu takip eden for ve
#if cumleleri baglamında değerlendirilmesinden kaynaklanan yeni bir liste olacaktir.


words = sorted(set(words))


classses = sorted(set(classses))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classses,open('classes.pkl','wb'))


training = []
output_empty = [0] * len(classses)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
       bag.append(1) if word in word_patterns else bag.append(0)


    output_row = list(output_empty)
    output_row[classses.index(document[1])] = 1
    training.append([bag,output_row])

random.shuffle(training)
training = np.array(training)


train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),),activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5',hist)
print("Done")





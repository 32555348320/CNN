
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical


(X_train, y_train), (X_test, y_test) = mnist.load_data() # MNIST veri setini yükleme aşaması


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) # Veriyi yeniden şekillendirme
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


X_train = X_train.astype('float32') # Veriyi float tipine çevirme ve normalize etme
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


y_train = to_categorical(y_train) # Etiketleri one-hot encode etme
y_test = to_categorical(y_test)


model = Sequential() # Modeli oluşturma
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Modeli derleme işlemi


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200) # Modeli eğitme aşaması


scores = model.evaluate(X_test, y_test, verbose=0) # Modeli değerlendirme aşaması
print(f'Test loss: {scores[0]}')
print(f'Test accuracy: {scores[1]}')


predictions = model.predict(X_test) # Tahminler


for i in range(10):
    print(f'Gerçek: {np.argmax(y_test[i])}, Tahmin: {np.argmax(predictions[i])}') # İlk 10 test verisinin gerçek ve tahmin edilen etiketlerini yazdırma


for i in range(10):
    plt.subplot(2, 5, i+1) # İlk 10 test görüntüsünü görselleştirme
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Tahmin: {np.argmax(predictions[i])}')
    plt.axis('off')
plt.show()


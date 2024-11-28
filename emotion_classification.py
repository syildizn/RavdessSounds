# Gerekli kütüphaneleri içe aktaralım
import librosa
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model

# Ayarlar
SAMPLE_RATE = 22050
SEGMENT_DURATION = 2
SAMPLES_PER_SEGMENT = 36000

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features_from_segment(audio_segment):
    mfccs = librosa.feature.mfcc(y=audio_segment, sr=SAMPLE_RATE, n_mfcc=40)
    return mfccs.T

def process_audio_file(file_path, label):
    audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
    total_samples = len(audio)
    segments = []
    
    for start in range(0, total_samples, SAMPLES_PER_SEGMENT):
        end = start + SAMPLES_PER_SEGMENT
        if end > total_samples:
            break
        segment = audio[start:end]
        features = extract_features_from_segment(segment)
        segments.append((features, label))
    return segments

data = []
labels = []
data_path = "/Users/syildizn/Documents/EmotionSoundsWorks/RavdessDataSet"

for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            emotion_code = file.split("-")[2]
            label = emotion_map.get(emotion_code, "unknown")
            segments = process_audio_file(file_path, label)
            for features, label in segments:
                data.append(features)
                labels.append(label)

# Zaman adımı uzunluklarını eşitlemek için padding
max_len = 80
X = pad_sequences(data, maxlen=max_len, padding='post', dtype='float32', truncating='post')

# Etiketleri sayısal verilere dönüştürme
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
y = to_categorical(y)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli Oluşturma
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(max_len, 40)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.4))

model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.4))

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(64))
model.add(Dropout(0.4))
model.add(Dense(y.shape[1], activation='softmax'))

# Modeli derleme
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model Grafiğini Çizdirme
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

# EarlyStopping ve ReduceLROnPlateau kullanımı
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)

# Modeli Eğitme
#history = model.fit(X_train, y_train, epochs=200, batch_size=32, 
                 #   validation_data=(X_test, y_test), callbacks=[early_stop, reduce_lr])
                 
history = model.fit(X_train, y_train, epochs=100, batch_size=32, 
                    validation_data=(X_test, y_test), callbacks=[reduce_lr])
                 

# Modeli Değerlendirme
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"Model doğruluğu: {accuracy}")
print("Sınıflandırma Raporu:")
print(classification_report(y_test_classes, y_pred_classes, target_names=label_encoder.classes_))

# Eğitim Sürecindeki Doğruluk ve Kayıp Grafikleri
plt.figure(figsize=(12, 4))

# Doğruluk Grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.legend()

# Kayıp Grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.legend()

plt.tight_layout()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
class_names = label_encoder.classes_

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Confusion Matrix - Sınıf Bazında Başarı')
plt.show()


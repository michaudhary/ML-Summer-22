import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models,utils
import pandas as pd
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow_io as tfio
#from helper import *
#importing all the helper fxn from helper.py which we will create later
import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

model = keras.models.load_model(os.path.abspath(os.path.dirname(__file__)))
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    if len(wav) < 48000:
        print('We cannot process audio less than three seconds long.')
        return
    centre = len(wav) // 2
    return wav[(centre-24000):(centre+24000)]
def extract_embedding(wav_data):
   #run YAMNet to extract embedding from the wav data 
  scores, embeddings, spectrogram = yamnet_model(wav_data)
  return embeddings
def predict(filename):
    classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'qawwali', 'reggae', 'rock']
    data = load_wav_16k_mono(filename)
    if data == None:
        return
    data = extract_embedding(data)
    data = np.array(data)
    data = data[0:6]
    data = data.reshape(1,6144)
    predicted_label = model.predict(data)
    classes_x = int(np.argmax(predicted_label,axis=1))
    prediction_class = classes[classes_x]
    return prediction_class
sns.set_theme(style="darkgrid")
sns.set()
st.title('Music Classifier')
def save_uploaded_file(uploaded_file):

    try:

        with open(os.path.join(os.getcwd(),uploaded_file.name),'wb') as f:

            f.write(uploaded_file.getbuffer())

        return 1    

    except:

        return 0
uploaded_file = st.file_uploader("Upload File")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes)
        prediction = predict(os.path.join('C:\\Users\\MAKTAB\\Documents', uploaded_file.name))
        os.remove(os.path.join(os.getcwd(),uploaded_file.name))
        st.text('We predict its genre to be ' + prediction)
    else:
        st.text('Sorry, we had some problems saving your file')

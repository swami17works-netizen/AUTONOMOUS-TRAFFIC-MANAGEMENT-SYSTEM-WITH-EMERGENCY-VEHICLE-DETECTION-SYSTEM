import numpy as np 
from tensorflow import keras
import tensorflow
from recorder import Recorder
import matplotlib.pyplot as plt
import scipy
import librosa
from librosa import display
from IPython.display import Audio
from scipy.fft import fft, fftfreq
import numpy as np
import cv2

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture('Traffic.mp4')
#cap = cv2.VideoCapture(1)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)


data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('keras_model.h5')
#model = tensorflow.keras.models.load_model('model_resnet50.h5')



def emergency():
    
   # r = Recorder()
    #r.record(5, output='out.wav')
    fname="sound_1.wav"
    samples, sam_rate=librosa.load(fname, sr=None, mono=True, offset=0.0, duration=None)


    Audio(fname)

    #plt.figure()
    #librosa.display.waveplot(y=samples, sr=sam_rate)
    #plt.xlabel("time secs")
    #plt.ylabel("Ampl")
    #plt.show()

    def fft_plot(audio, sam_rate):
        n = len(audio)
        T= 1/sam_rate
        yf = fft(audio)
        xf = fftfreq(n, T)#[:n//2]
        #fig, ax = plt.subplots()
        #plt.plot(xf, 2.0/n * np.abs(yf)) #np.abs(yf[0:n//2]))
        #plt.grid()
        #plt.xlabel("Freq")
        #plt.ylabel("mag")
        val=np.argmax(yf)
        #plt.show()
        return np.abs(xf[val])


    audio_freq = fft_plot(samples, sam_rate)
    freq = audio_freq.round()
    if freq in range(700,1500):
         cv2.putText(image,'Ambulance detecated',(450,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    
         print("it is a emergency vechicle")
    else:
        pass
    


amc = 0
# fir = 0
# pol = 0
while True:
    success, img = cap.read()
    image = img
    img = cv2.resize(img, (224, 224))
    img_res = np.asarray(img)
    normalized_image_array = (img_res.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    
    ambulance_index = 0
#     firengine_index = 1
#     policecar_index = 2
    traffic_index = 3
    
    # Define colors for each label
    ambulance_color = (0, 255, 0)  # Green
#     firengine_color = (0, 0, 255)   # Red
#     policecar_color = (255, 0, 0)   # Blue
    traffic_color = (0, 0, 0)        # Black
    
    # Update the accumulated counts
    if prediction[0][ambulance_index].round() == 1:
        amc += 1
        
    # If the count reaches a certain threshold, trigger the emergency function
    if amc == 20:
        amc = 0
        emergency()
    
    # Display detection information on the image
    if prediction[0][ambulance_index].round() == 1:
        cv2.putText(image, 'Ambulance detected', (450, 100), cv2.FONT_HERSHEY_COMPLEX, 1, ambulance_color, 2)
#     if prediction[0][firengine_index].round() == 1:
#         cv2.putText(image, 'Firengine detected', (450, 150), cv2.FONT_HERSHEY_COMPLEX, 1, firengine_color, 2)
#     if prediction[0][policecar_index].round() == 1:
#         cv2.putText(image, 'Policecar detected', (450, 200), cv2.FONT_HERSHEY_COMPLEX, 1, policecar_color, 2)
    if prediction[0][traffic_index].round() == 1:
        cv2.putText(image, 'Traffic detected', (450, 250), cv2.FONT_HERSHEY_COMPLEX, 1, traffic_color, 2)
    
    # Display accumulated counts
    text = f'Ambulance: {amc},'
    cv2.putText(image, text, (15, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("Result", image)
    
    key = cv2.waitKey(1) & 0xFF  # Capture the key pressed

    # If the 'ESC' key is pressed (27 is the ASCII code for 'ESC')
    if key == 27:
        cv2.destroyAllWindows() 
        break  # Exit the loop

from flask import Flask, render_template
from flask_socketio import SocketIO
import speech_recognition as sr
import threading
import os
import whisper

# Load the Whisper model
model = whisper.load_model("base")

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize recognizer
recognizer = sr.Recognizer()

# Variable to store the transcribed text
transcribed_text = None

# # Function to handle recording and transcription
# def record_audio():
#     global is_recording
#     with sr.Microphone() as source:
#         recognizer.adjust_for_ambient_noise(source)  # Adjust for noise
#         socketio.emit('status', {'message': 'Recording started... Speak now!'})
#         while is_recording:
#             try:
#                 audio = recognizer.listen(source, timeout=1)  # Listen for 1 second at a time
#                 if audio:
#                     # Use Google Web Speech API for transcription
#                     text = recognizer.recognize_google(audio)
#                     socketio.emit('status', {'message': f'Transcription: {text}'})
#             except sr.WaitTimeoutError:
#                 continue  # Continue listening
#             except sr.UnknownValueError:
#                 socketio.emit('status', {'message': 'Google Speech Recognition could not understand the audio.'})
#             except sr.RequestError as e:
#                 socketio.emit('status', {'message': f'Could not request results from Google Speech Recognition service; {e}'})



def record_audio():
    global is_recording
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)  # Adjust for noise
        socketio.emit('status', {'message': 'Recording started... Speak now!'})
        while is_recording:
            try:
                audio = recognizer.listen(source, timeout=1)  # Listen for 1 second at a time
                if audio:
                    # Save the audio to a temporary file
                    with open("temp_audio.wav", "wb") as f:
                        f.write(audio.get_wav_data())

                    # Transcribe the audio using Whisper
                    result = model.transcribe("temp_audio.wav")
                    text = result["text"]
                    socketio.emit('status', {'message': f'Transcription: {text}'})
            except sr.WaitTimeoutError:
                continue  # Continue listening
            except Exception as e:
                socketio.emit('status', {'message': f'An error occurred: {e}'})
            
# SocketIO event to start recording
@socketio.on('start_recording')
def handle_start_recording():
    global is_recording
    is_recording = True
    threading.Thread(target=record_audio).start()

# SocketIO event to stop recording
@socketio.on('stop_recording')
def handle_stop_recording():
    global is_recording
    is_recording = False
    socketio.emit('status', {'message': 'Recording stopped. Processing...'})

# Route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)
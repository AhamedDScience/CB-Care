import os
import wave
import pyaudio
import pygame
import speech_recognition as sr
from pyht import Client, TTSOptions, Format
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import numpy as np


# Google API Key setup
os.environ["GOOGLE_API_KEY"] = 'GOOGLE_API_KEY'

# Initialize Language Model and Embeddings
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.9)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="task_type_unspecified")

# Load Vector Database
vectordb_file_path = "faiss_index"
vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(score_threshold=0.7)

# Initialize TTS Client
client = Client("user id", "api code")

input_filename = "input_audio.wav"

# Function to transcribe audio to text
def separate_audio(audio_path):
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        return text
    except Exception as e:
        print(f"Error separate_audio audio: {e}")

# Prompt Template
prompt_template = """
Given the following context and a question, generate an answer based on this context only.
In the answer, try to provide as much text as possible from the "response" section in the source document context without making significant changes.
If the answer is not found in the context and is relevant to our business model boot camp, kindly state, "I don't have enough information to answer that question. Please refer to our team."
If the question is not relevant to our business model, kindly state, "Please ask only relevant questions."
Do not try to fabricate an answer.

CONTEXT: {context}

QUESTION: {question}
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Initialize RetrievalQA Chain
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, input_key="query", return_source_documents=True, chain_type_kwargs=chain_type_kwargs)

# Function to process audio and generate response
def process_audio(aud):
    text = separate_audio(aud)
    print("Transcribed Text:", text)
    
    response = chain({"query": text})
    answer = response['result'].strip()
    print("Generated Response:", answer)
    
    if answer:
        audio_filename = 'generated_audio5.mp3'
        options1 = TTSOptions(
            voice="s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json",
            sample_rate=24000,
            format=Format.FORMAT_MP3,
            speed=0.9,
        )
        with open("generated_audio5.mp3", "wb") as f:
            if response is not None:
                for chunk in client.tts(text=[answer], voice_engine="PlayHT2.0-turbo", options=options1):
                    f.write(chunk)
        print(f"Generated audio saved to {audio_filename}")
        return audio_filename
    return None

# Constants for audio recording
chunk = 1024
sample_format = pyaudio.paInt16
channels = 1
fs = 44100
silence_threshold = 800
silence_duration = 2

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = None
frames = []

def is_silent(data_chunk):
    as_ints = np.frombuffer(data_chunk, dtype=np.int16)
    return np.max(np.abs(as_ints)) < silence_threshold

def start_recording(input_device_index):
    print('Recording Started')
    global stream, frames
    frames = []
    stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True, input_device_index=input_device_index)
    record()

def record():
    global stream, frames
    silent_chunks = 0
    silent_for_long_enough = False
    
    while not silent_for_long_enough:
        data = stream.read(chunk)
        frames.append(data)
        
        if is_silent(data):
            silent_chunks += 1
            if (silent_chunks * chunk) / fs >= silence_duration:
                silent_for_long_enough = True
        else:
            silent_chunks = 0

    stop_recording(input_filename)
    process_and_play()

def stop_recording(filename):
    print('Recording Stopped')
    global stream, frames
    stream.stop_stream()
    stream.close()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))

def process_and_play():
    print('Generating the response!')
    output_filename = process_audio(input_filename)
    if output_filename:
        play_audio(output_filename)

def play_audio(filename):
    print('Playing the response')
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    print('Output generated!')
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()
    pygame.quit()
    print('Process complete')
    enable_recording_buttons()

def list_audio_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    devices = []
    
    for i in range(num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        devices.append((i, device_info.get('name')))
    
    p.terminate()
    return devices  


def enable_recording_buttons():
    start_button.config(state=tk.NORMAL)
    input_device_menu.config(state=tk.NORMAL)

def disable_recording_buttons():
    start_button.config(state=tk.DISABLED)
    input_device_menu.config(state=tk.DISABLED)

# Setup GUI
window = tk.Tk()
window.title("Video Recorder")
window.attributes('-fullscreen', True)

# Get screen width and height
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Create a style
style = ttk.Style()
style.theme_use('clam')

# Configure modern button style
style.configure('Modern.TButton', 
                font=('Helvetica', 12, 'bold'),
                foreground='white',
                background='#1E1E1E',
                padding=(20, 10),
                borderwidth=0,
                relief='flat')
style.map('Modern.TButton',
          foreground=[('active', 'white')],
          background=[('active', '#3E3E3E')])

# Configure other styles
style.configure('TLabel', font=('Helvetica', 12), background='#1E1E1E', foreground='white')
style.configure('TFrame', background='#1E1E1E')
style.configure('TCombobox', 
                selectbackground='#3E3E3E',
                fieldbackground='#3E3E3E',
                background='#3E3E3E',
                foreground='white',
                arrowcolor='white')
style.map('TCombobox', fieldbackground=[('readonly', '#3E3E3E')])


# Create a semi-transparent overlay
overlay = ttk.Frame(window, style='TFrame')
overlay.place(relwidth=1, relheight=1)

# Load and set the background image
bg_image = Image.open("cb2.png")
bg_image = bg_image.resize((screen_width, screen_height), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(overlay, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)

# Create a frame for the input device selection in the top-right corner
input_frame = ttk.Frame(overlay, style='TFrame')
input_frame.place(relx=0.98, rely=0.02, anchor='ne')



input_device_var = tk.StringVar(value="Select Device")
input_device_menu = ttk.Combobox(input_frame, textvariable=input_device_var, values=[f"{index}: {name}" for index, name in list_audio_devices()], 
                                 state="readonly", font=('Helvetica', 10), width=20)
input_device_menu.pack(side=tk.TOP, padx=5, pady=5)

# Create a frame for buttons in the bottom-right corner
button_frame = ttk.Frame(overlay, style='TFrame')
button_frame.place(relx=0.98, rely=0.98, anchor='se')

start_button = ttk.Button(button_frame, text="Start Recording", command=lambda: [disable_recording_buttons(), start_recording(int(input_device_var.get().split(':')[0]))],style='Modern.TButton')
start_button.pack(side=tk.TOP, padx=5, pady=5)

exit_button = ttk.Button(overlay, text="Exit", command=window.quit,style='Modern.TButton')
exit_button.place(relx=0.02, rely=0.02, anchor='nw')

# Start the Tkinter main loop
window.mainloop()

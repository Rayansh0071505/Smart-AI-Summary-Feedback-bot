import streamlit as st
import whisper
import numpy as np
from pydub import AudioSegment
import io
from dotenv import load_dotenv
import os
import openai

load_dotenv()

# Load the Whisper model for transcriptio
model = whisper.load_model("tiny")

# OpenAI API key setup
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key


def convert_audio_to_text(audio_file):
    # Convert audio to the required format (mono WAV)
    audio = AudioSegment.from_file_using_temporary_files(audio_file)
    audio = audio.set_channels(1).set_frame_rate(16000)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)
    
    # Load buffer into a numpy array and convert to float32
    samples = np.array(AudioSegment.from_wav(buffer).get_array_of_samples())
    if samples.dtype != np.float32:
        # Normalize short integers to float range -1.0 to 1.0
        samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
    
    # Transcribe the audio using Whisper
    result = model.transcribe(samples)
    return result["text"]

def analyze_conversation(text, query):
    # Use GPT-4 to summarize and provide feedback on the conversation
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Assuming use of gpt-3.5-turbo, adjust if different model needed
        messages=[
            {"role": "system", "content": f"Summarize this conversation and provide feedback in points, for your reference you are smart bot who will assist Amita Goyal (She generally used to have calls with client for the sales and feedback work, her work is to connect with client and make them understand what product she building and try to sell them or she used to connect with client regarding the feedback of products and help clients to resolve issue, also she gives demos to clients) , so your task is to help sales team and her which are selling the product, from the input text you received try to summarise the conversation/give feedback what client wants, also mention the changes that clients want based on conversation with our sales team guy about the product, analyze the conversation with client and try to give suggestion to Amita as a first person about what client wants and highlight important points of conversation/meeting:\n\n{text}"},
            {"role": "user", "content": query}
        ],
        temperature=0.3,
        max_tokens=1500
    )
    return response.choices[0].message['content']

st.title('Smart-Ai-Feedback-Summary-bot')
uploaded_file = st.file_uploader("Upload an audio or video file", type=['mp3', 'wav', 'mp4', 'avi', 'mov'])
user_query = st.text_input("Enter your query or instructions here:")

if uploaded_file is not None and user_query:
    with st.spinner('Processing...'):
        transcribed_text = convert_audio_to_text(uploaded_file)
        summary_and_feedback = analyze_conversation(transcribed_text, user_query)
        st.write("Summary and Feedback:")
        st.write(summary_and_feedback)
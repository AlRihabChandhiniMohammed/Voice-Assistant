import openai
import sounddevice as sd
import soundfile as sf
import pyttsx3
import tempfile
import os

# Set your OpenAI API key here or export as env var OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY") 

engine = pyttsx3.init()

def record_audio(duration=5, fs=16000):
    print("Speak now...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return audio.flatten(), fs

def save_audio_to_wav(audio, fs, filename):
    sf.write(filename, audio, fs)

def transcribe_audio_whisper(filename):
    with open(filename, "rb") as audio_file:
        print("Transcribing...")
        transcript = openai.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1"
        )
    return transcript['text']


def ask_chatgpt(prompt):
    print("ChatGPT is thinking...")
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    return response['choices'][0]['message']['content']

def speak_text(text):
    print("Assistant:", text)
    engine.say(text)
    engine.runAndWait()

def main():
    print("Starting voice assistant (say 'stop' or 'exit' to quit).")
    while True:
        audio, fs = record_audio(duration=5)  # record 5 seconds
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            save_audio_to_wav(audio, fs, tmpfile.name)
            try:
                text = transcribe_audio_whisper(tmpfile.name)
            except Exception as e:
                print(f"Transcription error: {e}")
                os.unlink(tmpfile.name)
                continue
            os.unlink(tmpfile.name)

        print("You said:", text)

        if text.strip().lower() in ["stop", "exit", "quit"]:
            speak_text("Goodbye!")
            break

        try:
            response = ask_chatgpt(text)
        except Exception as e:
            print(f"ChatGPT API error: {e}")
            continue

        speak_text(response)

if __name__ == "__main__":
    main()

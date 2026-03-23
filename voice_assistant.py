import os
import subprocess
import time
import re
import urllib.request
import json
import atexit

WHISPER_CLI = os.path.expanduser("~/whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL = os.path.expanduser("~/whisper.cpp/models/ggml-tiny.bin")

LLAMA_SERVER = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
LLAMA_MODEL = os.path.expanduser("~/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

PIPER_CLI = os.path.expanduser("~/piper/build/piper")
PIPER_MODEL = os.path.expanduser("~/piper/voices/en_US-lessac-medium.onnx")

llama_process = None

def cleanup():
    global llama_process
    if llama_process:
        print("\n[🧹] Shutting down AI models...")
        llama_process.terminate()

atexit.register(cleanup)

def start_llama_server():
    global llama_process
    print("[⚙️] Booting up TinyLlama Server... (This takes 10-15 seconds ONCE)")
    
    # Run the server in the background, suppressing its logs
    devnull = open(os.devnull, 'w')
    llama_process = subprocess.Popen(
        [LLAMA_SERVER, "-m", LLAMA_MODEL, "--port", "8080", "-c", "2048"],
        stdout=devnull, stderr=devnull
    )
    
    # Wait for the server to be ready
    for _ in range(30):
        try:
            req = urllib.request.Request("http://127.0.0.1:8080/health")
            with urllib.request.urlopen(req, timeout=1) as response:
                if response.status == 200:
                    print("[✓] AI Model loaded into memory securely!")
                    return
        except:
            pass
        time.sleep(1)
    
    print("[!] Warning: Llama server may not have started properly.")

def record_audio(duration=5, output_file="user.wav"):
    print(f"\n[🎙️] Listening for {duration} seconds... Please speak!")
    devnull = open(os.devnull, 'w')
    subprocess.run(["arecord", "-D", "plughw:1,0", "-d", str(duration), "-f", "S16_LE", "-c", "1", "-r", "16000", output_file],
                   stdout=devnull, stderr=devnull)
    devnull.close()
    print("[✓] Recording finished.")

def transcribe_audio(audio_file="user.wav"):
    print("[🧠] Transcribing with Whisper...")
    result = subprocess.run([WHISPER_CLI, "-m", WHISPER_MODEL, "-f", audio_file, "-nt"],
                            capture_output=True, text=True)
    
    transcript = result.stdout.strip()
    transcript = re.sub(r'\[.*?\]', '', transcript).strip()
    transcript = re.sub(r'\(.*?\)', '', transcript).strip()
    
    # Filter out common Whisper silence hallucinations
    hallucinations = ["hello.", "hello", "thank you.", "thank you", "thanks for watching.", "thanks.", "bye."]
    if transcript.lower() in hallucinations:
        print("[!] Ignored standard background static (Whisper hallucination).")
        return ""
    
    if len(transcript) < 2:
        return ""
    
    print(f"[User]: {transcript}")
    return transcript

import random

# Load predefined responses
PREDEFINED_RESPONSES = {}
try:
    with open(os.path.expanduser("~/Desktop/test/reposees.json"), "r") as f:
        # The user's file is slightly malformed at the end with a dot, let's be careful
        content = f.read().strip()
        if content.endswith("."):
            content = content[:-1]
        PREDEFINED_RESPONSES = json.loads(content)
except Exception as e:
    print(f"[!] Error loading reposees.json: {e}")

import datetime

def get_current_time_ist():
    # Get current UTC time and adjust for IST (+5:30)
    # The server system time is already in IST (+05:30) based on meta
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

def log_interaction(query, response, source):
    # Log interactions to help "learn" in the future
    try:
        with open("interaction_logs.txt", "a") as f:
            f.write(f"[{datetime.datetime.now()}] [{source}] Q: {query} | A: {response}\n")
    except:
        pass

def get_json_response(text):
    text = text.lower().strip()
    words = re.findall(r'\b\w+\b', text)
    
    # Priority 1: Dynamic Time (Whole word check)
    if "time" in words or "clock" in words:
        time_str = get_current_time_ist()
        return f"Currently, it is {time_str} in India, dude." if "india" in words else f"It's {time_str}."

    # Priority 2: Precise JSON Knowledge
    for key, data in PREDEFINED_RESPONSES.items():
        if "inputs" in data:
            for trigger in data["inputs"]:
                # Check for exact trigger or if trigger is a meaningful part of the text
                t = trigger.lower()
                if t in text and (len(t) > 3 or t == text):
                    if data["responses"]:
                        return random.choice(data["responses"])
    return None

def generate_response(prompt_text):
    # First, check the JSON knowledge base (includes Time)
    json_resp = get_json_response(prompt_text)
    if json_resp:
        log_interaction(prompt_text, json_resp, "JSON")
        print(f"[📋] Using Knowledge Base for: {prompt_text}")
        print(f"[Assistant]: {json_resp}")
        return json_resp

    print("[🤖] Thinking with TinyLlama...")
    # System prompt: keep it extremely minimal and conversational
    full_prompt = f"<|system|>\nYou are a helpful assistant. Reply concisely in one sentence. Do not lecture.</s>\n<|user|>\n{prompt_text}</s>\n<|assistant|>\n"
    
    data = json.dumps({
        "prompt": full_prompt,
        "n_predict": 64,
        "temperature": 0.5,
        "stop": ["</s>", "<|user|>", "\n"]
    }).encode('utf-8')
    
    req = urllib.request.Request("http://127.0.0.1:8080/completion", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode('utf-8'))
            text = result.get("content", "").strip()
            text = text.replace("AI:", "").strip()
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if lines:
                text = lines[0]
            
            log_interaction(prompt_text, text, "LLM")
            print(f"[Assistant]: {text}")
            return text
    except Exception as e:
        print(f"[!] Llama error: {str(e)}")
        return "Sorry, I had an error thinking."

def speak_text(text, output_file="response.wav"):
    print("[🗣️] Synthesizing speech with Piper...")
    piper_process = subprocess.Popen([PIPER_CLI, "-m", PIPER_MODEL, "-f", output_file],
                                     stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    piper_process.communicate(input=text.encode('utf-8'))
    
    print("[🔈] Playing response...")
    devnull = open(os.devnull, 'w')
    subprocess.run(["aplay", "-D", "plughw:3,0", output_file], stdout=devnull, stderr=devnull)
    devnull.close()

def main():
    print("=====================================================")
    print("  🚀 Medha's Voice Assistant Started! (Ctrl+C to stop) ")
    print("=====================================================")
    
    start_llama_server()
    
    # Initial Greeting
    greeting = "Hi, this is Medha, your AI voice assistant. How can I help you today?"
    print(f"\n[👋] {greeting}")
    speak_text(greeting)
    
    try:
        while True:
            # Continuous mode
            record_audio(duration=5, output_file="user.wav")
            transcript = transcribe_audio("user.wav")
            
            if not transcript or len(transcript) < 2:
                continue
                
            response = generate_response(transcript)
            
            if response:
                speak_text(response, output_file="response.wav")
            
            print("\nREADY (listening again in 1s)...")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[👋] Exiting Voice Assistant... Goodbye!")
        # atexit handles cleanup

if __name__ == "__main__":
    main()

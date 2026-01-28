import torch
from transformers import pipeline
import os

# This line stops that "ReadTimeout" network error from popping up
os.environ["TRANSFORMERS_OFFLINE"] = "0" 

TEST_FILE = "001_Daudi_trudsins.mp3"
MODEL_ID = "language-and-voice-lab/whisper-large-icelandic-62640-steps-967h"

def run_safe_test():
    print(f"--- Starting SAFE MODE Test ---")
    print("Goal: Get real Icelandic text (no stuttering)")
    
    # We switch to 'cpu' because the Mac GPU is causing the '!!m!!!' error
    device = "cpu"
    
    print(f"Loading Model: {MODEL_ID}...")
    
    # We use float32 (Full Precision) for perfect accuracy
    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_ID,
        device=device,
        torch_dtype=torch.float32, 
        chunk_length_s=30
    )

    print(f"Transcribing {TEST_FILE}...")
    print("This will be slower on CPU. Please wait 3-5 minutes...")
    
    result = pipe(
        TEST_FILE,
        batch_size=1, # Lower batch size is more stable for testing
        generate_kwargs={"language": "icelandic", "task": "transcribe"}
    )

    print("\n--- TRANSCRIPTION PREVIEW ---")
    if result["text"].strip():
        print(result["text"][:500])
    else:
        print("Empty result - Something is still wrong with the model load.")
    print("\n--- END PREVIEW ---")

    with open("safe_test_output.txt", "w", encoding="utf-8") as f:
        f.write(result["text"])
    
    print(f"Full text saved to: safe_test_output.txt")

if __name__ == "__main__":
    run_safe_test()
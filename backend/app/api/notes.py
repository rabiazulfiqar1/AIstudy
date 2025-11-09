# def generate_summary(text: str):
#     """Generate summary from text"""
#     load_models()  # Ensure summarizer is loaded
#     summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
#     return summary[0]["summary_text"]

import os
import librosa
import torch
import tempfile
import requests
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from typing import Optional
from dotenv import load_dotenv
from transformers import pipeline

# Load the .env file from the backend folder
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

HF_TOKEN = os.getenv("HF_TOKEN")  # Your Hugging Face Access Token
print(HF_TOKEN)
HF_SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_SUMMARIZATION_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Globals for models
processor = None
model = None

CHUNK_SIZE = 30 * 16000  # 30 sec chunks

router = APIRouter()


def load_models():
    """Lazy-load Whisper models only when first request comes in"""
    global processor, model
    if processor is None or model is None:
        print("Loading Whisper model...")
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        # summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        # summarizer = pipeline(
        #     "summarization",
        #     model="t5-small"
        # )
        print("Whisper model loaded.")


def load_audio_file(file_path: str):
    """Load audio at 16kHz and return a torch tensor"""
    speech_array, sr = librosa.load(file_path, sr=16000)
    return torch.from_numpy(speech_array)


def get_chunks(speech_array, chunk_size):
    """Yield sequential chunks of fixed length"""
    for i in range(0, len(speech_array), chunk_size):
        yield speech_array[i : i + chunk_size]

        
def chunk_text_by_sentence(text, max_length=1000):
    """Split text into chunks of roughly max_length ending at sentence boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        if end < len(text):
            # backtrack to last sentence-ending punctuation
            last_punct = max(text.rfind(p, start, end) for p in ".!?")
            if last_punct != -1 and last_punct > start:
                end = last_punct + 1  # include punctuation
        chunks.append(text[start:end].strip())
        start = end
    return chunks


def generate_transcript(file_path: str):
    """Transcribe long audio by chunking"""
    load_models()  # Ensure models are loaded
    full_transcript = []
    speech_array = load_audio_file(file_path)
    for chunk in get_chunks(speech_array, CHUNK_SIZE):
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
        predicted_ids = model.generate(inputs.input_features)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print(text)
        full_transcript.append(text)

    final_text = " ".join(full_transcript)
    return final_text

def generate_summary(text: str, chunk_max_length=3000, chunk_summary_max=650):
    """Aggressively summarize long transcript in chunks and then combine."""
    if HF_TOKEN is None:
        raise HTTPException(status_code=500, detail="HF_TOKEN not set in environment variables.")

    # Step 1: Chunk transcript
    chunks = chunk_text_by_sentence(text, max_length=chunk_max_length)
    chunk_summaries = []

    for chunk in chunks:
        payload = {
            "inputs": chunk,
            "parameters": {
                "max_new_tokens": chunk_summary_max,  
                "min_new_tokens": 20,
            }
        }
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
        print("HF status code:", response.status_code)
        print("HF response:", response.text)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        data = response.json()
        if isinstance(data, list) and "summary_text" in data[0]:
            chunk_summaries.append(data[0]["summary_text"])
        else:
            raise HTTPException(status_code=500, detail=f"Unexpected response: {data}")

    # Step 2: Combine all chunk summaries
    combined_summary = " ".join(chunk_summaries)

    # Step 3: Summarize the combined summary again for final concise output
    # payload = {
    #     "inputs": f"Summarize the following text concisely but include all important points:\n\n{combined_summary}",
    #     "parameters": {
    #         "max_new_tokens": 2000,  # final concise summary
    #     }
    # }
    
    # print(combined_summary)
    # response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
    # if response.status_code != 200:
    #     raise HTTPException(status_code=response.status_code, detail=response.text)

    # data = response.json()
    # if isinstance(data, list) and "summary_text" in data[0]:
    #     final_summary = data[0]["summary_text"]
    # else:
    #     raise HTTPException(status_code=500, detail=f"Unexpected response: {data}")

    # print("Final concise summary:\n", final_summary)
    # return final_summary
    return combined_summary

@router.post("/get_transcript")
async def get_transcript(
    file: Optional[UploadFile] = File(None),
    link: Optional[str] = Form(None),
):
    print("Get transcript")
    if not file and not link:
        raise HTTPException(status_code=400, detail="Please provide a file or link")

    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            transcript = generate_transcript(tmp_path)
        finally:
            os.remove(tmp_path)
    elif link:
        raise HTTPException(status_code=501, detail="Link transcription not yet supported")

    return {"transcript": transcript}


@router.post("/get_summary")
async def get_summary(
    file: Optional[UploadFile] = File(None),
    link: Optional[str] = Form(None),
):
    if not file and not link:
        raise HTTPException(status_code=400, detail="Please provide a file or link")

    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            transcript = generate_transcript(tmp_path)
        finally:
            os.remove(tmp_path)
    elif link:
        raise HTTPException(status_code=501, detail="Link transcription not yet supported")

    summary = generate_summary(transcript)
    return {"summary": summary}

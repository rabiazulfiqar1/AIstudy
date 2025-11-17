import os
import librosa
import torch
import tempfile
import requests
import subprocess
import yt_dlp
import hashlib
import pickle #noqa
from pathlib import Path
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from supabase import create_client, Client
from typing import Optional, List, Dict
from dotenv import load_dotenv
import json
from datetime import datetime
from datetime import timedelta

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

HF_TOKEN = os.getenv("HF_TOKEN")
HF_SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
HF_NOTE_GENERATION_MODEL = "google/flan-t5-base"
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

# Supabase configuration
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Use the new router endpoint
HF_SUMMARY_API = f"https://router.huggingface.co/hf-inference/models/{HF_SUMMARIZATION_MODEL}"
HF_NOTES_API = f"https://router.huggingface.co/hf-inference/models/{HF_NOTE_GENERATION_MODEL}" 
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Cache directory
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

# Globals
processor = None
model = None

CHUNK_DURATION = 30
SAMPLE_RATE = 16000 #1sec of audio: 16000 samples note: more samples, more accurate
CHUNK_SIZE = CHUNK_DURATION * SAMPLE_RATE

router = APIRouter()


# ============== CACHING UTILITIES ==============

def get_file_hash(file_content: bytes) -> str:
    """Generate hash from file content"""
    return hashlib.sha256(file_content).hexdigest()


def get_link_hash(url: str) -> str:
    """Generate hash from URL"""
    return hashlib.sha256(url.encode()).hexdigest()


def get_cache_path(cache_key: str, cache_type: str) -> Path:
    """
    Get cache file path for a given key and type
    cache_type: 'transcript', 'summary', 'notes', 'full'
    """
    return CACHE_DIR / f"{cache_key}_{cache_type}.pkl"


# def save_to_cache_local(cache_key: str, cache_type: str, data: dict):
#     """Save processed data to cache"""
#     cache_path = get_cache_path(cache_key, cache_type)
#     try:
#         with open(cache_path, 'wb') as f:
#             pickle.dump(data, f)
#         print(f"✅ Cached to: {cache_path}")
#     except Exception as e:
#         print(f"❌ Cache save failed: {e}")
        
async def save_to_cache(cache_key: str, cache_type: str, data: dict):
    """
    Save processed data to Supabase cache
    Uses upsert to handle both insert and update cases
    """
    try:
        cache_entry = {
            "cache_key": cache_key,
            "cache_type": cache_type,
            "data": json.dumps(data),  # Store as JSON string
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Upsert: insert or update if exists
        result = supabase.table("video_cache").upsert(
            cache_entry,
            on_conflict="cache_key,cache_type"
        ).execute()
        
        print(f"✅ Cached to Supabase: {cache_key}_{cache_type}")
        return result
    except Exception as e:
        print(f"❌ Supabase cache save failed: {e}")
        # Don't raise - caching failure shouldn't break the main flow
        return None


# def load_from_cache_local(cache_key: str, cache_type: str) -> Optional[dict]:
#     """Load processed data from cache"""
#     cache_path = get_cache_path(cache_key, cache_type)
    
#     if cache_path.exists():
#         try:
#             with open(cache_path, 'rb') as f:
#                 data = pickle.load(f)
#             print(f"✅ Cache HIT: {cache_path}")
#             return data
#         except Exception as e:
#             print(f"❌ Cache load failed: {e}")
#             return None
    
#     print(f"❌ Cache MISS: {cache_path}")
#     return None

async def load_from_cache(cache_key: str, cache_type: str) -> Optional[dict]:
    """Load processed data from Supabase cache"""
    try:
        result = supabase.table("video_cache").select("*").eq(
            "cache_key", cache_key
        ).eq(
            "cache_type", cache_type
        ).execute()
        
        if result.data and len(result.data) > 0:
            cache_entry = result.data[0]
            data = json.loads(cache_entry["data"])
            
            # Update last_accessed timestamp
            supabase.table("video_cache").update({
                "last_accessed": datetime.utcnow().isoformat()
            }).eq("id", cache_entry["id"]).execute()
            
            print(f"✅ Cache HIT from Supabase: {cache_key}_{cache_type}")
            return data
        
        print(f"❌ Cache MISS: {cache_key}_{cache_type}")
        return None
    except Exception as e:
        print(f"❌ Supabase cache load failed: {e}")
        return None

# def clear_old_cache_local(max_age_days: int = 7):
#     """Delete cache files older than max_age_days"""
#     import time
#     current_time = time.time()
#     max_age_seconds = max_age_days * 86400 #24(hrs)*60(mins)*60(secs) = 86400 seconds in a day
    
#     deleted_count = 0
#     for cache_file in CACHE_DIR.glob("*.pkl"):
#         if current_time - cache_file.stat().st_mtime > max_age_seconds:
#             cache_file.unlink()
#             deleted_count += 1
    
#     if deleted_count > 0:
#         print(f"🗑️ Deleted {deleted_count} old cache files")


async def clear_old_cache(max_age_days: int = 7):
    """Delete cache entries older than max_age_days"""
    try:
        cutoff_date = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()
        
        # Delete entries not accessed recently
        result = supabase.table("video_cache").delete().lt(
            "updated_at", cutoff_date
        ).execute()
        
        deleted_count = len(result.data) if result.data else 0
        
        if deleted_count > 0:
            print(f"🗑️ Deleted {deleted_count} old cache entries from Supabase")
        
        return deleted_count
    except Exception as e:
        print(f"❌ Failed to clear old cache: {e}")
        return 0

async def clear_cache_by_key(cache_key: str):
    """Delete all cache entries for a specific key"""
    try:
        result = supabase.table("video_cache").delete().eq(
            "cache_key", cache_key
        ).execute()
        
        deleted_count = len(result.data) if result.data else 0
        print(f"🗑️ Deleted {deleted_count} entries for key: {cache_key}")
        return deleted_count
    except Exception as e:
        print(f"❌ Failed to clear cache by key: {e}")
        return 0
    
async def get_all_cache_keys(limit: int = 100) -> List[str]:
    """Get list of all unique cache keys"""
    try:
        result = supabase.table("video_cache").select("cache_key").limit(limit).execute()
        
        if result.data:
            # Get unique keys
            keys = list(set([entry["cache_key"] for entry in result.data]))
            return keys
        return []
    except Exception as e:
        print(f"❌ Failed to get cache keys: {e}")
        return []

# ============== EXISTING FUNCTIONS ==============

def load_models():
    """Lazy-load Whisper models"""
    global processor, model
    if processor is None or model is None:
        print("Loading Whisper model...")
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en")
        print("Whisper model loaded.")


def download_video_from_link(url: str) -> str:
    """Download video from YouTube or other platforms"""
    print(f"Downloading video from: {url}")
    
    output_dir = tempfile.gettempdir()
    output_template = os.path.join(output_dir, '%(id)s.%(ext)s')
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', #fall back to any best format if mp4 not available
        'outtmpl': output_template,
        'quiet': False, #false means show progress in terminal.
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info['id']
            ext = info['ext']
            downloaded_file = os.path.join(output_dir, f"{video_id}.{ext}")
            
            print(f"Downloaded to: {downloaded_file}")
            return downloaded_file
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


def extract_audio_from_video(video_path: str) -> str:
    """Extract audio using ffmpeg"""
    print(f"Extracting audio from: {video_path}")
    
    audio_path = video_path.rsplit('.', 1)[0] + '.wav'
    
    try:
        subprocess.run([
            "ffmpeg", "-i", video_path, #input file
            "-ar", "16000", #sets audio sample rate to 16 kHz
            "-ac", "1", #converts audio to mono (1 channel).
            "-y", #overwrite the output file if it already exists.
            audio_path #output file path.
        ], check=True, capture_output=True) 
#check=True raises CalledProcessError on failure, capture_output=True captures stdout/stderr (so it won’t flood the console)
        
        print(f"Audio extracted to: {audio_path}")
        return audio_path
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e.stderr.decode()}")


def load_audio_file(file_path: str):
    """Load audio at 16kHz"""
    speech_array, sr = librosa.load(file_path, sr=SAMPLE_RATE) #a NumPy array of audio samples. Each sample is a float number representing the amplitude of the sound wave at a specific time.
    return torch.from_numpy(speech_array) #converts it into a PyTorch tensor for Whisper.


def get_chunks_with_offset(speech_array, chunk_size, sample_rate):
    """Yield chunks with time offsets"""
    for i in range(0, len(speech_array), chunk_size):
        offset_seconds = i / sample_rate # converts the sample index into actual time in seconds. 48000 samples / 16000 samples/second = 3 seconds
        chunk = speech_array[i : i + chunk_size]
        yield chunk, offset_seconds


def generate_transcript_with_timestamps(file_path: str) -> List[Dict]:
    """Transcribe audio with timestamps"""
    load_models()
    segments = []
    speech_array = load_audio_file(file_path)
    total_duration = len(speech_array) / SAMPLE_RATE
    
    print(f"Transcribing {total_duration:.2f} seconds of audio...")
    
    for chunk, offset in get_chunks_with_offset(speech_array, CHUNK_SIZE, SAMPLE_RATE):
        inputs = processor(             #Converts the raw audio chunk into model-ready input.
            chunk, 
            sampling_rate=SAMPLE_RATE, 
            return_tensors="pt", 
            padding=True #pads the input to the longest sequence in the batch
        )
        
        predicted_ids = model.generate(inputs.input_features, return_timestamps=True)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0] #batch_decode converts token IDs into human-readable text.
        
        chunk_duration = len(chunk) / SAMPLE_RATE
        end_time = min(offset + chunk_duration, total_duration)
        
        if text.strip():
            segment = {
                "start": round(offset, 2),
                "end": round(end_time, 2),
                "text": text.strip()
            }
            segments.append(segment)
            print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {text[:50]}...")
    
    return segments


def segments_to_full_text(segments: List[Dict]) -> str:
    """Convert segments to full transcript"""
    return " ".join([seg["text"] for seg in segments])


def chunk_text_smart(text: str, max_length: int = 1000) -> List[str]:
    """Split text at sentence boundaries"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        if end < len(text):
            last_punct = max((text.rfind(p, start, end) for p in ".!?"), default=-1)
            if last_punct > start:
                end = last_punct + 1
        chunks.append(text[start:end].strip())
        start = end
    return chunks


def call_hf_inference(api_url: str, text: str, max_tokens: int = 500) -> str:
    """Call HuggingFace Inference API"""
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN not set")
    
    payload = {
        "inputs": text,
        "parameters": {
            "max_new_tokens": max_tokens,
            "min_new_tokens": 50,
            "do_sample": False
        }
    }
    
    response = requests.post(api_url, headers=HEADERS, json=payload)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"HF API error: {response.text}")
    
    data = response.json() # data is Python list containing one dict.
    
    if isinstance(data, list) and len(data) > 0:
        if "summary_text" in data[0]:
            return data[0]["summary_text"]
        elif "generated_text" in data[0]:
            return data[0]["generated_text"]
    
    return str(data)


def generate_summary_hf(text: str, max_summary_length: int = 500) -> str:
    """Generate summary using HF API"""
    chunks = chunk_text_smart(text, max_length=3000)
    chunk_summaries = []
    
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        summary = call_hf_inference(HF_SUMMARY_API, chunk, max_tokens=max_summary_length)
        chunk_summaries.append(summary)
    
    combined = " ".join(chunk_summaries)
    
    if len(combined) > 2000 and len(chunks) > 1:
        combined = call_hf_inference(HF_SUMMARY_API, combined, max_tokens=max_summary_length)
    
    return combined

def call_groq_inference(prompt: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct", max_tokens: int = 300) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0
    }

    resp = requests.post(GROQ_API_URL, headers=headers, json=payload)
    
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Groq API error: {resp.text}")

    data = resp.json()
    
    try:
        # returns the assistant's message
        return data["choices"][0]["message"]["content"]
    except Exception as e: #noqa
        raise HTTPException(status_code=500, detail=f"Unexpected Groq response: {data}")

def generate_smart_notes_hf(segments: List[Dict]) -> str:
    """Generate structured notes using HF API"""
    print("Generating smart notes...")
    notes = ["# Lecture Notes\n"]
    
    section_duration = 300  # 5 minutes
    current_section = []
    section_start = 0
    section_num = 1
    
    for seg in segments:
        if seg["start"] - section_start > section_duration and current_section:
            section_text = " ".join([s["text"] for s in current_section])
            timestamp_range = f"[{format_timestamp(section_start)} - {format_timestamp(current_section[-1]['end'])}]"
            
            prompt = f"""You are a teacher’s assistant. Read this lecture transcript and generate concise, **high-quality bullet points**. Only include main ideas, formulas, and examples. Ignore filler words.

Transcript: {section_text}

Key Points:"""
            
            try:
                # section_notes = call_hf_inference(HF_NOTES_API, prompt, max_tokens=300)
                section_notes = call_groq_inference(prompt, max_tokens=300)
                notes.append(f"\n## Section {section_num} {timestamp_range}\n")
                notes.append(f"{section_notes}\n")
            except Exception as e:
                print(f"Error generating notes: {e}")
                notes.append(f"\n## Section {section_num} {timestamp_range}\n")
                sentences = section_text.split('. ')
                for sentence in sentences[:5]:
                    if len(sentence.strip()) > 20:
                        notes.append(f"- {sentence.strip()}\n")
            
            current_section = []
            section_start = seg["start"]
            section_num += 1
        
        current_section.append(seg)
    
    # Process last section
    if current_section:
        section_text = " ".join([s["text"] for s in current_section])
        timestamp_range = f"[{format_timestamp(section_start)} - {format_timestamp(current_section[-1]['end'])}]"
        
        try:
            section_notes = call_hf_inference(HF_NOTES_API, f"Extract key points: {section_text}", max_tokens=300)
            notes.append(f"\n## Section {section_num} {timestamp_range}\n")
            notes.append(f"{section_notes}\n")
        except: #noqa
            notes.append(f"\n## Section {section_num} {timestamp_range}\n")
            sentences = section_text.split('. ')
            for sentence in sentences[:5]:
                if len(sentence.strip()) > 20:
                    notes.append(f"- {sentence.strip()}\n")
    
    return "".join(notes)

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS or MM:SS"""
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


# ============== CACHED API ENDPOINTS ==============

@router.post("/get_transcript")
async def get_transcript(
    file: Optional[UploadFile] = File(None),
    link: Optional[str] = Form(None),
):
    """Get transcript with caching"""
    if not file and not link:
        raise HTTPException(status_code=400, detail="Provide file or link")
    
    # Generate cache key
    if file:
        file_content = await file.read()
        cache_key = get_file_hash(file_content)
    else:
        cache_key = get_link_hash(link)
    
    # Check cache first
    cached = await load_from_cache(cache_key, 'transcript')
    if cached:
        return {**cached, "from_cache": True}
    
    # Process if not cached
    video_path = None
    audio_path = None
    
    try:
        if file:
            suffix = Path(file.filename).suffix if file.filename else ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_content)
                video_path = tmp.name
            
            if suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                audio_path = extract_audio_from_video(video_path)
            else:
                audio_path = video_path
        
        elif link:
            video_path = download_video_from_link(link)
            audio_path = extract_audio_from_video(video_path)
        
        segments = generate_transcript_with_timestamps(audio_path)
        full_text = segments_to_full_text(segments)
        
        result = {
            "segments": segments,
            "full_text": full_text,
            "duration": segments[-1]["end"] if segments else 0
        }
        
        # Save to cache
        await save_to_cache(cache_key, 'transcript', result)
        
        return {**result, "from_cache": False}
    
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        if audio_path and audio_path != video_path and os.path.exists(audio_path):
            os.remove(audio_path)


@router.post("/get_summary")
async def get_summary(
    file: Optional[UploadFile] = File(None),
    link: Optional[str] = Form(None),
    max_length: int = Form(500)
):
    """Get summary with caching"""
    if not file and not link:
        raise HTTPException(status_code=400, detail="Provide file or link")
    
    # Generate cache key
    if file:
        file_content = await file.read()
        cache_key = get_file_hash(file_content)
    else:
        cache_key = get_link_hash(link)
    
    # Check cache
    cached = await load_from_cache(cache_key, 'summary')
    if cached:
        return {**cached, "from_cache": True}
    
    video_path = None
    audio_path = None
    
    try:
        if file:
            suffix = Path(file.filename).suffix if file.filename else ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_content)
                video_path = tmp.name
            
            if suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                audio_path = extract_audio_from_video(video_path)
            else:
                audio_path = video_path
        
        elif link:
            video_path = download_video_from_link(link)
            audio_path = extract_audio_from_video(video_path)
        
        # Check if transcript is cached
        transcript_cached = await load_from_cache(cache_key, 'transcript')
        if transcript_cached:
            segments = transcript_cached['segments']
            full_text = transcript_cached['full_text']
        else:
            segments = generate_transcript_with_timestamps(audio_path)
            full_text = segments_to_full_text(segments)
        
        try:
            summary = generate_summary_hf(full_text, max_summary_length=max_length)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")
        
        result = {
            "summary": summary,
            "original_length": len(full_text),
            "summary_length": len(summary),
            "duration": segments[-1]["end"] if segments else 0
        }
        
        await save_to_cache(cache_key, 'summary', result)
        
        return {**result, "from_cache": False}
    
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        if audio_path and audio_path != video_path and os.path.exists(audio_path):
            os.remove(audio_path)


@router.post("/process_video")
async def process_video(
    file: Optional[UploadFile] = File(None),
    link: Optional[str] = Form(None),
):
    """Full processing with caching"""
    if not file and not link:
        raise HTTPException(status_code=400, detail="Provide file or link")
    
    # Generate cache key
    if file:
        file_content = await file.read()
        cache_key = get_file_hash(file_content)
    else:
        cache_key = get_link_hash(link)
    
    # Check full cache
    cached = await load_from_cache(cache_key, 'full')
    if cached:
        return {**cached, "from_cache": True}
    
    video_path = None
    audio_path = None
    
    try:
        if file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(file_content)
                video_path = tmp.name
        elif link:
            video_path = download_video_from_link(link)
        
        audio_path = extract_audio_from_video(video_path)
        
        # Check if transcript cached
        transcript_cached = await load_from_cache(cache_key, 'transcript')
        if transcript_cached:
            segments = transcript_cached['segments']
            full_text = transcript_cached['full_text']
        else:
            segments = generate_transcript_with_timestamps(audio_path)
            full_text = segments_to_full_text(segments)
            await save_to_cache(cache_key, 'transcript', {
                'segments': segments,
                'full_text': full_text,
                'duration': segments[-1]["end"] if segments else 0
            })
        
        notes = generate_smart_notes_hf(segments)
        summary = generate_summary_hf(full_text, max_summary_length=300)
        
        result = {
            "segments": segments,
            "notes": notes,
            "summary": summary,
            "duration": segments[-1]["end"] if segments else 0,
            "format": "markdown"
        }
        
        await save_to_cache(cache_key, 'full', result)
        
        return {**result, "from_cache": False}
    
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        if audio_path and audio_path != video_path and os.path.exists(audio_path):
            os.remove(audio_path)


@router.post("/generate_notes")
async def generate_notes(
    file: Optional[UploadFile] = File(None),
    link: Optional[str] = Form(None),
):
    """Generate notes with caching"""
    if not file and not link:
        raise HTTPException(status_code=400, detail="Provide file or link")
    
    if file:
        file_content = await file.read()
        cache_key = get_file_hash(file_content)
    else:
        cache_key = get_link_hash(link)
    
    cached = await load_from_cache(cache_key, 'notes')
    if cached:
        return {**cached, "from_cache": True}
    
    video_path = None
    audio_path = None
    
    try:
        if file:
            suffix = Path(file.filename).suffix if file.filename else ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_content)
                video_path = tmp.name
            
            if suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                audio_path = extract_audio_from_video(video_path)
            else:
                audio_path = video_path
        elif link:
            video_path = download_video_from_link(link)
            audio_path = extract_audio_from_video(video_path)
        
        # Reuse transcript if cached
        transcript_cached = await load_from_cache(cache_key, 'transcript')
        if transcript_cached:
            segments = transcript_cached['segments']
        else:
            segments = generate_transcript_with_timestamps(audio_path)
        
        notes = generate_smart_notes_hf(segments)
        
        result = {
            "notes": notes,
            "segments": segments,
            "format": "markdown",
            "duration": segments[-1]["end"] if segments else 0
        }
        
        await save_to_cache(cache_key, 'notes', result)
        
        return {**result, "from_cache": False}
    
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        if audio_path and audio_path != video_path and os.path.exists(audio_path):
            os.remove(audio_path)

# ============== CACHE MANAGEMENT ENDPOINTS ==============

@router.post("/clear_cache")
async def clear_cache(max_age_days: int = 7):
    """Clear old cache entries from Supabase"""
    deleted_count = await clear_old_cache(max_age_days)
    return {
        "message": f"Cleared cache older than {max_age_days} days",
        "deleted_count": deleted_count
    }


@router.delete("/cache/{cache_key}")
async def delete_cache_entry(cache_key: str):
    """Delete all cache entries for a specific key"""
    deleted_count = await clear_cache_by_key(cache_key)
    return {
        "message": f"Deleted cache entries for key: {cache_key}",
        "deleted_count": deleted_count
    }


@router.get("/cache/keys")
async def list_cache_keys(limit: int = 100):
    """List all cache keys"""
    keys = await get_all_cache_keys(limit)
    return {
        "keys": keys,
        "count": len(keys),
        "limit": limit
    }
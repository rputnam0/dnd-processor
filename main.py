import os
import zipfile
import shutil
import whisperx
import torch
import ffmpeg
import pandas as pd
from flask import Flask
from google.cloud import storage
from google.cloud import secretmanager
from datetime import timedelta

# --- Configuration ---
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
secret_client = secretmanager.SecretManagerServiceClient()
storage_client = storage.Client()
app = Flask(__name__)

# --- Helper Functions ---
def get_secret(secret_id, version_id="latest"):
    """Fetches a secret from Google Cloud Secret Manager."""
    name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/{version_id}"
    response = secret_client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

def format_time(seconds):
    return str(timedelta(seconds=int(seconds))).zfill(8)

def get_speaker_label_from_zip(filename, speaker_mapping):
    base_name = os.path.basename(filename)
    parts = base_name.split('-')
    key_part = os.path.splitext(parts[-1] if len(parts) > 1 else base_name)[0]
    return speaker_mapping.get(key_part, base_name)

def preprocess_audio_for_diarization(input_file):
    """Converts any audio file to the 16kHz mono WAV format required by WhisperX."""
    print(f"Preprocessing {input_file} for diarization...")
    output_file = "/tmp/preprocessed_audio.wav"
    try:
        ffmpeg.input(input_file).output(
            output_file, acodec='pcm_s16le', ac=1, ar='16k'
        ).run(overwrite_output=True, quiet=True)
        print("Preprocessing successful.")
        return output_file
    except ffmpeg.Error as e:
        print(f"Error preprocessing audio: {e.stderr.decode()}")
        raise

# --- Main Processing Logic ---
@app.route("/", methods=["POST"])
def process_audio_file():
    print("Received trigger. Scanning for new files...")
    bucket = storage_client.bucket(BUCKET_NAME)

  
    uploaded_blobs = bucket.list_blobs(prefix="audio-uploads/")
    transcript_blobs = bucket.list_blobs(prefix="transcripts/", delimiter="/")

    # Create a dictionary mapping base filenames (e.g., "Session 31") to full filenames ("Session 31.zip")
    uploaded_files_map = {
        os.path.splitext(os.path.basename(b.name))[0]: os.path.basename(b.name)
        for b in uploaded_blobs if not b.name.endswith('/')
    }

    # Create a set of already processed sessions by looking at the folder names in "transcripts/"
    processed_session_folders = {folder.split('/')[1] for folder in transcript_blobs.prefixes}

    # Find the first file that hasn't been processed yet
    unprocessed_base_name = next(
        (base_name for base_name in sorted(uploaded_files_map.keys()) if base_name not in processed_session_folders), 
        None
    )

    if not unprocessed_base_name:
        print("No new files to process.")
        return ("No new files to process.", 200)

    # Get the full filename from the map to process
    file_to_process = uploaded_files_map[unprocessed_base_name]
    print(f"Found new file to process: {file_to_process}")
    
    # --- Route to the correct processing function based on file type ---
    if file_to_process.lower().endswith('.zip'):
        return process_zip_file(file_to_process, bucket)
    
    elif file_to_process.lower().endswith(('.m4a', '.mp3', '.flac', '.wav', '.ogg')):
        return process_single_file(file_to_process, bucket)
    
    else:
        # To prevent retries on unsupported files, create an empty marker file.
        base_name = os.path.splitext(file_to_process)[0]
        bucket.blob(f"transcripts/{base_name}/unsupported_file_type.txt").upload_from_string("")
        return (f"Unsupported file type: {file_to_process}", 400)

def process_zip_file(file_name, bucket):
    print(f"Processing multi-track ZIP file: {file_name}")
    local_download_path = f"/tmp/{file_name}"
    local_extract_dir = "/tmp/unzipped_audio"
    os.makedirs(local_extract_dir, exist_ok=True)

    # Download, extract, and transcribe...
    blob = bucket.blob(f"audio-uploads/{file_name}")
    blob.download_to_filename(local_download_path)
    
    audio_files = []
    with zipfile.ZipFile(local_download_path, 'r') as zip_ref:
        zip_ref.extractall(local_extract_dir)
    for root, _, files in os.walk(local_extract_dir):
        for f in files:
            if f.lower().endswith(('.flac', '.ogg')):
                audio_files.append(os.path.join(root, f))
    
    device = 'cuda'
    model = whisperx.load_model("large-v3", device, compute_type='float16')
    speaker_mapping = {'bfschmity_0': 'Kaladen Shash', 'jessev567890_0': 'Zariel Torgan', 'kinglizard7958_0': 'Leopold Magnus', 'joeeeenathan_0': 'Dungeon Master', 'travisaurus6985_0': 'Cyrus Schwert', 'traceritops_0': 'Cletus Cobbington'}

    all_segments = []
    for audio_file in audio_files:
        speaker = get_speaker_label_from_zip(audio_file, speaker_mapping)
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=32)
        for segment in result["segments"]:
            all_segments.append((segment['start'], f"{speaker} {format_time(segment['start'])} {segment['text'].strip()}"))
    
    consolidated = sorted(all_segments, key=lambda x: x[0])
    base_filename = os.path.splitext(file_name)[0]
    local_transcript_path = f"/tmp/{base_filename}.txt"
    with open(local_transcript_path, 'w') as f:
        for _, text in consolidated:
            f.write(f"{text}\n")

    transcript_blob = bucket.blob(f"transcripts/{base_filename}/{base_filename}.txt")
    transcript_blob.upload_from_filename(local_transcript_path)
    print(f"Successfully uploaded transcript for {file_name}.")
    
   
    blob.delete()
    print(f"Cleaned up original audio file from GCS: {file_name}")

    shutil.rmtree(local_extract_dir)
    os.remove(local_download_path)
    os.remove(local_transcript_path)
    return (f"Successfully processed ZIP file {file_name}", 200)

def process_single_file(file_name, bucket):
    print(f"Processing single-track audio file: {file_name}")
    local_download_path = f"/tmp/{file_name}"

    blob = bucket.blob(f"audio-uploads/{file_name}")
    blob.download_to_filename(local_download_path)
    
    processed_audio_path = preprocess_audio_for_diarization(local_download_path)
    device = 'cuda'
    model = whisperx.load_model("large-v3", device, compute_type='float16')
    audio = whisperx.load_audio(processed_audio_path)
    result = model.transcribe(audio, batch_size=16)

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    hf_token = get_secret("hf-token")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    base_filename = os.path.splitext(file_name)[0]
    local_transcript_path = f"/tmp/{base_filename}.txt"
    with open(local_transcript_path, 'w') as f:
        for segment in result["segments"]:
            speaker = segment.get('speaker', 'UNKNOWN')
            text = segment['text'].strip()
            f.write(f"{format_time(segment['start'])}: {speaker}: {text}\n")
            
    transcript_blob = bucket.blob(f"transcripts/{base_filename}/{base_filename}.txt")
    transcript_blob.upload_from_filename(local_transcript_path)
    print(f"Successfully uploaded transcript for {file_name}.")

    # --- NEW: Cleanup audio file from GCS ---
    blob.delete()
    print(f"Cleaned up original audio file from GCS: {file_name}")

    os.remove(local_download_path)
    os.remove(processed_audio_path)
    os.remove(local_transcript_path)
    return (f"Successfully processed single file {file_name}", 200)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
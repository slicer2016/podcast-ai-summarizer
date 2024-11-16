import torch
from speechbrain.inference import EncoderClassifier
from openai import OpenAI
import librosa
import numpy as np
from pydub import AudioSegment
from sklearn.cluster import KMeans
from pathlib import Path
import tempfile
import time
import requests
from config import OPENAI_API_KEY, ELEVENLABS_API_KEY

class PodcastProcessor:
    def __init__(self, start_time_minutes=5):
        """Initialize processor"""
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.elevenlabs_api_key = ELEVENLABS_API_KEY
        
        self.embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        )
        
        self.start_time_ms = start_time_minutes * 60 * 1000
        self.temp_dir = Path(tempfile.mkdtemp())

    def _split_audio(self, audio_segment, max_chunk_size_mb=20):
        """Split audio into processable chunks"""
        chunk_length_ms = 60000
        chunks = []
        
        while True:
            test_chunk = audio_segment[:chunk_length_ms]
            temp_path = self.temp_dir / "test_chunk.wav"
            test_chunk.export(temp_path, format='wav')
            
            if temp_path.stat().st_size / (1024 * 1024) < max_chunk_size_mb:
                break
            
            chunk_length_ms = int(chunk_length_ms * 0.8)

        total_length_ms = len(audio_segment)
        for start_ms in range(0, total_length_ms, chunk_length_ms):
            end_ms = min(start_ms + chunk_length_ms, total_length_ms)
            chunks.append(audio_segment[start_ms:end_ms])
            
        return chunks

    def _transcribe_segment(self, segment_path):
        """Transcribe audio using Whisper API"""
        try:
            with open(segment_path, "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            return transcript.text
        except Exception as e:
            if "413" in str(e):
                audio_segment = AudioSegment.from_wav(segment_path)
                chunks = self._split_audio(audio_segment)
                
                full_transcript = []
                for i, chunk in enumerate(chunks):
                    chunk_path = self.temp_dir / f"chunk_{i}.wav"
                    chunk.export(chunk_path, format='wav')
                    
                    with open(chunk_path, "rb") as chunk_file:
                        chunk_transcript = self.openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=chunk_file
                        )
                    full_transcript.append(chunk_transcript.text)
                
                return " ".join(full_transcript)
            else:
                raise

    def _summarize_text(self, text):
        """Summarize text using GPT"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Summarize the following text in 1-2 concise sentences while maintaining key information:"},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error summarizing text: {str(e)}")
            return text

    def _generate_voice(self, text, voice_id):
        """Generate voice using ElevenLabs"""
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "xi-api-key": self.elevenlabs_api_key,
                "Content-Type": "application/json"
            }
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }
            
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code != 200:
                raise Exception(f"API request failed with status {response.status_code}")
            
            temp_path = self.temp_dir / "temp_voice.mp3"
            with open(temp_path, "wb") as f:
                f.write(response.content)
            
            return AudioSegment.from_mp3(temp_path)
            
        except Exception as e:
            print(f"Error generating voice: {str(e)}")
            raise

    def _perform_diarization(self, audio_path, window_seconds=3.0):
        """Perform speaker diarization"""
        audio, sr = librosa.load(audio_path, sr=16000)
        audio = torch.FloatTensor(audio)
        
        window_size = int(window_seconds * sr)
        hop_size = window_size // 2
        
        embeddings = []
        timestamps = []
        
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            emb = self.embedding_model.encode_batch(window.unsqueeze(0))
            embeddings.append(emb.squeeze().cpu().numpy())
            timestamps.append(i / sr)
        
        embeddings = np.array(embeddings)
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        segments = []
        current_speaker = labels[0]
        start_time = timestamps[0]
        
        for i in range(1, len(labels)):
            if labels[i] != current_speaker:
                segments.append({
                    'speaker': int(current_speaker),
                    'start': start_time,
                    'end': timestamps[i]
                })
                current_speaker = labels[i]
                start_time = timestamps[i]
        
        segments.append({
            'speaker': int(current_speaker),
            'start': start_time,
            'end': timestamps[-1]
        })
        
        speaker_times = {0: 0, 1: 0}
        for segment in segments:
            speaker_times[segment['speaker']] += segment['end'] - segment['start']
        
        host_speaker = 0 if speaker_times[0] < speaker_times[1] else 1
        
        return segments, host_speaker

    def process_file(self, input_path, output_path, voice_id, min_duration_seconds=15):
        """Process audio file"""
        audio = AudioSegment.from_file(input_path)
        segments, host_speaker = self._perform_diarization(str(input_path))
        
        final_audio = AudioSegment.empty()
        current_position = 0
        
        if self.start_time_ms > 0:
            final_audio += audio[:self.start_time_ms]
            current_position = self.start_time_ms
        
        for segment in segments:
            start_ms = int(segment['start'] * 1000)
            end_ms = int(segment['end'] * 1000)
            
            if end_ms <= self.start_time_ms:
                continue
            
            if start_ms < self.start_time_ms:
                start_ms = self.start_time_ms
            
            if start_ms > current_position:
                final_audio += audio[current_position:start_ms]
            
            segment_duration = (end_ms - start_ms) / 1000
            
            if segment['speaker'] == host_speaker and segment_duration >= min_duration_seconds:
                segment_audio = audio[start_ms:end_ms]
                temp_path = self.temp_dir / f"segment_{start_ms}.wav"
                segment_audio.export(temp_path, format='wav')
                
                transcript = self._transcribe_segment(temp_path)
                summary = self._summarize_text(transcript)
                new_audio = self._generate_voice(summary, voice_id)
                
                final_audio += new_audio
            else:
                final_audio += audio[start_ms:end_ms]
            
            current_position = end_ms
        
        if current_position < len(audio):
            final_audio += audio[current_position:]
        
        final_audio.export(output_path, format="mp3")
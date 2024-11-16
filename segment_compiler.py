from pydub import AudioSegment
import pandas as pd
import requests
from pathlib import Path
import tempfile

class SegmentCompiler:
    def __init__(self, elevenlabs_api_key):
        """Initialize the audio segment compiler
        
        Args:
            elevenlabs_api_key (str): API key for ElevenLabs voice synthesis
        """
        self.elevenlabs_api_key = elevenlabs_api_key
        self.temp_dir = Path(tempfile.mkdtemp())

    def create_comparison(self, csv_path, original_path, edited_path, output_path, 
                         segment_ids, voice_id):
        """Create an audio comparison file from original and edited segments
        
        Args:
            csv_path (str): Path to CSV containing segment information
            original_path (str): Path to original audio file
            edited_path (str): Path to edited audio file
            output_path (str): Path for output comparison file
            segment_ids (list): List of segment IDs to include
            voice_id (str): ElevenLabs voice ID for announcements
        """
        # Load audio files and segment data
        df = pd.read_csv(csv_path)
        original_audio = AudioSegment.from_file(original_path)
        edited_audio = AudioSegment.from_file(edited_path)
        
        # Create standard announcements
        intro_text = "Audio comparison demonstration. Original vs processed segments:"
        section_a = "Original segment:"
        section_b = "Processed segment:"
        
        intro_audio = self._generate_voice(intro_text, voice_id)
        orig_announce = self._generate_voice(section_a, voice_id)
        edit_announce = self._generate_voice(section_b, voice_id)
        
        # Create silence segments
        long_silence = AudioSegment.silent(duration=2000)
        short_silence = AudioSegment.silent(duration=1000)
        
        # Start with intro
        final_audio = intro_audio + long_silence
        
        # Process each segment
        for segment_id in segment_ids:
            segment = df[df['segment_id'] == segment_id].iloc[0]
            
            # Add original segment
            final_audio += orig_announce + short_silence
            orig_start = int(segment['orig_start_time'] * 1000)
            orig_end = int(segment['orig_end_time'] * 1000)
            orig_segment = original_audio[orig_start:orig_end].fade_out(2000)
            final_audio += orig_segment + long_silence
            
            # Add edited segment
            final_audio += edit_announce + short_silence
            edit_start = int(segment['edit_start_time'] * 1000)
            edit_end = int(segment['edit_end_time'] * 1000)
            edit_segment = edited_audio[edit_start:edit_end].fade_out(2000)
            final_audio += edit_segment + long_silence
        
        # Export final compilation
        final_audio.export(output_path, format="mp3")
        print(f"Created comparison file: {output_path}")

    def _generate_voice(self, text, voice_id):
        """Generate voice using ElevenLabs API
        
        Args:
            text (str): Text to convert to speech
            voice_id (str): ElevenLabs voice ID
            
        Returns:
            AudioSegment: Generated audio segment
        """
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

    def __del__(self):
        """Cleanup temporary files on object deletion"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

def main():
    """Example usage"""
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description='Create audio comparisons')
    parser.add_argument('--api-key', required=True, help='ElevenLabs API key')
    parser.add_argument('--csv', required=True, help='Path to segments CSV')
    parser.add_argument('--original', required=True, help='Path to original audio')
    parser.add_argument('--edited', required=True, help='Path to edited audio')
    parser.add_argument('--output', required=True, help='Path for output file')
    parser.add_argument('--voice-id', required=True, help='ElevenLabs voice ID')
    parser.add_argument('--segments', required=True, help='Comma-separated segment IDs')
    
    args = parser.parse_args()
    
    compiler = SegmentCompiler(args.api_key)
    segment_ids = [int(x.strip()) for x in args.segments.split(',')]
    
    compiler.create_comparison(
        csv_path=args.csv,
        original_path=args.original,
        edited_path=args.edited,
        output_path=args.output,
        segment_ids=segment_ids,
        voice_id=args.voice_id
    )

if __name__ == "__main__":
    main()
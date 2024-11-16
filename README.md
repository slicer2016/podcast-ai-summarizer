# 🎙️ Podcast AI Summarizer

<!-- Centered badges -->
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered tool for processing and summarizing long-form audio content.

[Key Features](#key-features) •
[Installation](#installation) •
[Quick Start](#quick-start) •
[Documentation](#documentation) •
[License](#license)

---

🚀 Key Features

- 🎯 **Speaker Diarization**: Automatically identifies different speakers
- 🗣️ **Speech-to-Text**: Accurate transcription using OpenAI Whisper
- ✨ **Smart Summarization**: GPT-powered content summarization
- 🔊 **Voice Synthesis**: Natural voice generation with ElevenLabs
- 📊 **Comparison Tool**: Generate before/after comparisons

📋 Requirements

- Python 3.8+
- OpenAI API key
- ElevenLabs API key

💻 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/podcast-ai-summarizer.git

# Navigate to directory
cd podcast-ai-summarizer

# Install requirements
pip install -r requirements.txt
```

🏃‍♂️ Quick Start

    1. Process an Audio File

```python
from podcast_processor import PodcastProcessor

processor = PodcastProcessor(openai_api_key="your-key", elevenlabs_api_key="your-key")
processor.process_file(
    input_path="podcast.mp3",
    output_path="processed.mp3",
    voice_id="your-voice-id"
)
```

2. Create Comparisons

```python
from segment_compiler import SegmentCompiler

compiler = SegmentCompiler(elevenlabs_api_key="your-key")
compiler.create_comparison(
    csv_path="segments.csv",
    original_path="original.mp3",
    edited_path="edited.mp3",
    output_path="comparison.mp3",
    segment_ids=[1, 2, 3],
    voice_id="your-voice-id"
)
```

📖 Documentation

CSV Format

Your segments CSV should follow this format:
```csv
segment_id,orig_start_time,orig_end_time,edit_start_time,edit_end_time
1,120.5,145.2,90.3,100.1
2,360.8,390.4,250.2,265.5
```

Command Line Usage

```bash
python segment_compiler.py \
  --api-key YOUR_ELEVENLABS_API_KEY \
  --csv segments.csv \
  --original original.mp3 \
  --edited processed.mp3 \
  --output comparison.mp3 \
  --voice-id YOUR_VOICE_ID \
  --segments "1,2,3"
```

Arguments

| Argument | Description |
|----------|-------------|
| `--api-key` | Your ElevenLabs API key |
| `--csv` | Path to segments CSV file |
| `--original` | Path to original audio |
| `--edited` | Path to processed audio |
| `--output` | Output file path |
| `--voice-id` | ElevenLabs voice ID |
| `--segments` | Comma-separated segment IDs |

⚠️ Important Notes

- This tool is for research and personal use only
- Only process content you have rights to modify
- Requires accurate segment timing information

🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with ❤️ for podcast enthusiasts

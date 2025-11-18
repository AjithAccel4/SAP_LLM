# Multi-Modal Invoice Processing Examples

This directory contains examples demonstrating SAP_LLM's advanced multi-modal capabilities for ultra-enterprise invoice processing scenarios.

## Features

### 1. Video Document Processing
- **Keyframe Extraction**: Extract keyframes using PySceneDetect
- **Temporal Consistency**: Validate data consistency across frames
- **Scene Detection**: Automatic scene change detection
- **Supported Formats**: MP4, AVI, MOV, MKV, WEBM
- **Performance**: Process 30fps video in <5 seconds

### 2. Audio Processing
- **Speech-to-Text**: Whisper-based transcription
- **Multi-Language**: Support for 10+ languages (en, es, fr, de, it, pt, nl, pl, ru, zh, ja, ko, ar, hi, tr)
- **Speaker Diarization**: Identify and separate multiple speakers (optional)
- **Confidence Scoring**: Per-segment confidence scores
- **Target WER**: <5% Word Error Rate

### 3. Complex Table Extraction
- **TableTransformer**: State-of-the-art table detection and structure recognition
- **Nested Tables**: Detect and extract hierarchical table structures
- **Merged Cells**: Handle complex merged cells (horizontal and vertical)
- **Multi-Page Tables**: Track tables spanning multiple pages
- **Table Validation**: Automatic totals and subtotals verification
- **Target Accuracy**: ≥90% on nested tables

### 4. Multi-Modal Fusion
- **Attention-Based Fusion**: Cross-modal attention between vision, text, audio, and video
- **Confidence Weighting**: Dynamic modality weighting based on quality
- **Consistency Checking**: Cross-modal consistency validation
- **Temporal Encoding**: Temporal relationships for video sequences
- **Target Accuracy**: ≥95% with multiple modalities

## Installation

Install required dependencies:

```bash
# Basic dependencies (already in requirements.txt)
pip install -r requirements.txt

# For video processing
pip install scenedetect[opencv]

# For audio processing
pip install openai-whisper librosa soundfile pydub

# For speaker diarization (optional)
pip install pyannote.audio

# For table extraction
pip install timm
pip install 'git+https://github.com/facebookresearch/detectron2.git@main'
```

### HuggingFace Token for Diarization

If using speaker diarization, you need a HuggingFace token:

1. Create account at https://huggingface.co
2. Accept terms for pyannote models at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Generate token at https://huggingface.co/settings/tokens
4. Pass token to AudioProcessor:

```python
processor = AudioProcessor(
    enable_diarization=True,
    huggingface_token="your_token_here"
)
```

## Usage

### Basic Example

```python
from sap_llm.stages.preprocessing import PreprocessingStage
from sap_llm.models.audio_processor import AudioProcessor
from sap_llm.models.table_extractor import TableExtractor
from sap_llm.models.multimodal_fusion import AdvancedMultiModalFusion

# Initialize components
preprocessing = PreprocessingStage()
audio_processor = AudioProcessor(model_size="base")
table_extractor = TableExtractor()
fusion = AdvancedMultiModalFusion()
```

### Video Invoice Processing

```python
# Process video invoice
result = preprocessing.process({
    "document_path": "invoice_video.mp4",
    "document_id": "vid_001"
})

print(f"Extracted {len(result['keyframes'])} keyframes")
print(f"Temporal consistency: {result['temporal_consistency']:.2f}")
print(f"FPS: {result['fps']}, Duration: {result['duration']:.1f}s")
```

### Audio Invoice Processing

```python
# Process audio invoice
result = audio_processor.extract_invoice_data(
    audio_path="invoice_audio.wav",
    language="en"  # or None for auto-detect
)

print(f"Transcription: {result['transcription']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Language: {result['language']}")
print(f"Extracted entities: {result['extracted_entities']}")
```

### Complex Table Extraction

```python
from PIL import Image

# Load image
image = Image.open("invoice_with_tables.pdf")

# Get OCR results
ocr_result = preprocessing.process({"document_path": "invoice_with_tables.pdf"})

# Extract tables
tables = table_extractor.extract_tables(
    image=ocr_result['pages'][0],
    page_number=1,
    ocr_results=ocr_result['ocr_results'][0]
)

for table in tables:
    print(f"Table {table.table_id}: {table.rows}x{table.cols}")
    print(f"  Headers: {table.headers}")
    print(f"  Nested: {table.is_nested}")
    print(f"  Confidence: {table.confidence:.2f}")

    # Validate table
    validation = table_extractor.validate_table(table)
    print(f"  Valid: {validation['is_valid']}")
```

### Multi-Modal Fusion

```python
import torch

# Prepare features from different modalities
# (In production, use actual vision encoder and language decoder)
vision_features = torch.randn(1, 50, 768)
text_features = torch.randn(1, 50, 768)
audio_features = torch.randn(1, 50, 768)
video_features = torch.randn(1, 10, 768)

# Fuse modalities
result = fusion(
    vision_features=vision_features,
    text_features=text_features,
    audio_features=audio_features,
    video_features=video_features,
)

print(f"Modalities used: {result['modality_names']}")
print(f"Modality weights: {result['modality_weights']}")
print(f"Consistency scores: {result['consistency_scores']}")
```

## Examples

### Run Complete Example

```bash
python examples/multimodal/multimodal_processing_example.py
```

This demonstrates:
- Video invoice processing with keyframe extraction
- Audio invoice transcription with Whisper
- Complex table extraction with TableTransformer
- Multi-modal fusion with all modalities

## Performance Targets

| Feature | Target | Actual |
|---------|--------|--------|
| Video Processing (30fps) | <5s | Depends on scene count |
| Audio WER | <5% | Varies by language/quality |
| Table Extraction Accuracy | ≥90% | Model dependent |
| Multi-Modal Fusion Accuracy | ≥95% | With all modalities |
| End-to-End Latency | <3s | Per document |

## File Structure

```
examples/multimodal/
├── README.md                           # This file
├── multimodal_processing_example.py   # Complete example
├── test_multimodal.py                 # Unit tests
└── sample_data/                       # Sample files (add your own)
    ├── sample_invoice.mp4             # Sample video
    ├── sample_invoice.wav             # Sample audio
    └── sample_invoice.pdf             # Sample PDF
```

## Troubleshooting

### Video Processing Issues

**Error: PySceneDetect not available**
```bash
pip install scenedetect[opencv]
```

**Error: No scenes detected**
- Adjust `scene_threshold` in config (default: 27.0)
- Lower value = more sensitive (more scenes)
- Higher value = less sensitive (fewer scenes)

### Audio Processing Issues

**Error: Whisper not available**
```bash
pip install openai-whisper
```

**Poor transcription quality**
- Try larger model: `model_size="medium"` or `"large"`
- Ensure audio quality is good (16kHz+, low noise)
- Enable audio enhancement: `enhance_audio=True`

**High WER (Word Error Rate)**
- Check audio quality and format
- Try specifying language explicitly
- Use larger Whisper model

### Table Extraction Issues

**Error: Transformers not available**
```bash
pip install transformers timm
```

**Tables not detected**
- Adjust `confidence_threshold` (default: 0.7)
- Check image quality and resolution
- Ensure tables have clear borders

**Merged cells not detected**
- Tables should have visible cell boundaries
- Complex layouts may need manual post-processing

### GPU Issues

**CUDA out of memory**
- Reduce batch size
- Use smaller models (Whisper "base" instead of "large")
- Process fewer keyframes for videos

**No GPU available**
- Models will run on CPU (slower but functional)
- Consider cloud GPU instances for production

## Advanced Configuration

### Preprocessing Configuration

```python
from types import SimpleNamespace

config = SimpleNamespace(
    ocr_engine="easyocr",           # or "tesseract", "trocr"
    target_dpi=300,
    languages=["en", "es", "fr"],
    enable_video_processing=True,
    max_keyframes=30,
    scene_threshold=27.0,
)

preprocessing = PreprocessingStage(config)
```

### Audio Configuration

```python
audio_processor = AudioProcessor(
    model_size="medium",            # tiny, base, small, medium, large
    enable_diarization=True,        # Enable speaker separation
    huggingface_token="token",      # Required for diarization
)
```

### Table Extraction Configuration

```python
table_extractor = TableExtractor(
    model_name="microsoft/table-transformer-detection",
    structure_model="microsoft/table-transformer-structure-recognition",
    confidence_threshold=0.7,
)
```

### Multi-Modal Fusion Configuration

```python
fusion = AdvancedMultiModalFusion(
    fusion_dim=768,
    num_heads=32,
    num_layers=2,
    dropout=0.1,
)
```

## References

- **PySceneDetect**: https://github.com/Breakthrough/PySceneDetect
- **Whisper**: https://github.com/openai/whisper
- **TableTransformer**: https://github.com/microsoft/table-transformer
- **Pyannote**: https://github.com/pyannote/pyannote-audio

## Support

For issues or questions:
1. Check this README
2. Review example code
3. Check logs for error messages
4. Open issue on GitHub

## License

See main repository LICENSE file.

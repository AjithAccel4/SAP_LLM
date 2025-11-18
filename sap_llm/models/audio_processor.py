"""
Audio Processing Module with Whisper Integration.

Extracts invoice data from audio descriptions using:
- OpenAI Whisper for speech-to-text transcription
- Multi-language support (10+ languages)
- Speaker diarization for multi-party conversations
- Confidence scoring for audio extractions
- Audio enhancement and noise reduction

Target Metrics:
- WER (Word Error Rate): <5%
- Multi-language support: 10+ languages
- Latency: <2 seconds for 60-second audio
- Speaker diarization accuracy: ≥85%
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import tempfile
from pathlib import Path

import torch
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not available. Install with: pip install openai-whisper")

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logger.warning("Pyannote not available. Install with: pip install pyannote.audio")


class AudioProcessor:
    """
    Audio processing module for invoice data extraction.

    Features:
    - Speech-to-text with Whisper
    - Multi-language support
    - Speaker diarization
    - Audio enhancement
    - Confidence scoring
    """

    SUPPORTED_LANGUAGES = [
        "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "zh",
        "ja", "ko", "ar", "hi", "tr"
    ]

    SUPPORTED_FORMATS = [".wav", ".mp3", ".m4a", ".ogg", ".flac"]

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        enable_diarization: bool = False,
        huggingface_token: Optional[str] = None,
    ):
        """
        Initialize audio processor.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cuda/cpu). Auto-detects if None
            enable_diarization: Enable speaker diarization
            huggingface_token: HuggingFace token for pyannote models
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("Whisper is required. Install with: pip install openai-whisper")

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_size = model_size
        self.enable_diarization = enable_diarization

        logger.info(f"Initializing AudioProcessor with Whisper model: {model_size} on {device}")

        # Load Whisper model
        try:
            self.whisper_model = whisper.load_model(model_size, device=device)
            logger.info(f"Whisper model '{model_size}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

        # Load diarization pipeline if enabled
        self.diarization_pipeline = None
        if enable_diarization:
            if not PYANNOTE_AVAILABLE:
                logger.warning("Diarization requested but pyannote not available")
            else:
                try:
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=huggingface_token
                    )
                    if torch.cuda.is_available():
                        self.diarization_pipeline.to(torch.device("cuda"))
                    logger.info("Speaker diarization pipeline loaded")
                except Exception as e:
                    logger.warning(f"Failed to load diarization pipeline: {e}")

    def process_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        enhance_audio: bool = True,
    ) -> Dict[str, Any]:
        """
        Process audio file and extract text.

        Args:
            audio_path: Path to audio file
            language: Language code (auto-detect if None)
            task: "transcribe" or "translate" (to English)
            enhance_audio: Apply audio enhancement

        Returns:
            {
                "text": str,  # Full transcription
                "segments": List[Dict],  # Time-segmented transcription
                "language": str,  # Detected/specified language
                "confidence": float,  # Average confidence score
                "speakers": Optional[List[Dict]],  # Speaker diarization results
                "duration": float,  # Audio duration in seconds
            }
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if audio_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {audio_path.suffix}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

        logger.info(f"Processing audio file: {audio_path}")

        # Convert to WAV if needed and apply enhancement
        processed_audio_path = self._prepare_audio(audio_path, enhance_audio)

        # Get audio duration
        audio_data, sample_rate = librosa.load(str(processed_audio_path), sr=16000)
        duration = len(audio_data) / sample_rate

        # Transcribe with Whisper
        transcription_result = self._transcribe(
            processed_audio_path,
            language=language,
            task=task,
        )

        # Speaker diarization if enabled
        speakers = None
        if self.enable_diarization and self.diarization_pipeline is not None:
            speakers = self._diarize(processed_audio_path)
            # Merge speaker info with transcription segments
            transcription_result["segments"] = self._merge_speakers_and_segments(
                transcription_result["segments"],
                speakers
            )

        # Clean up temporary file if created
        if processed_audio_path != audio_path:
            processed_audio_path.unlink()

        result = {
            "text": transcription_result["text"],
            "segments": transcription_result["segments"],
            "language": transcription_result["language"],
            "confidence": self._calculate_confidence(transcription_result),
            "speakers": speakers,
            "duration": duration,
        }

        logger.info(
            f"Audio processing complete: {duration:.1f}s, "
            f"language={result['language']}, confidence={result['confidence']:.2f}"
        )

        return result

    def _prepare_audio(
        self,
        audio_path: Path,
        enhance: bool = True
    ) -> Path:
        """
        Prepare audio for processing.

        - Convert to WAV if needed
        - Resample to 16kHz
        - Apply noise reduction if enabled

        Args:
            audio_path: Input audio path
            enhance: Apply audio enhancement

        Returns:
            Path to prepared audio file
        """
        # Convert to WAV if not already
        if audio_path.suffix.lower() != ".wav":
            logger.debug(f"Converting {audio_path.suffix} to WAV")
            audio = AudioSegment.from_file(str(audio_path))

            # Create temp WAV file
            temp_wav = tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            )
            audio.export(temp_wav.name, format="wav")
            audio_path = Path(temp_wav.name)

        # Load audio
        audio_data, sample_rate = librosa.load(str(audio_path), sr=None)

        # Resample to 16kHz if needed (Whisper's expected rate)
        if sample_rate != 16000:
            logger.debug(f"Resampling from {sample_rate}Hz to 16000Hz")
            audio_data = librosa.resample(
                audio_data,
                orig_sr=sample_rate,
                target_sr=16000
            )
            sample_rate = 16000

        # Apply enhancement if requested
        if enhance:
            audio_data = self._enhance_audio(audio_data, sample_rate)

        # Save enhanced audio
        enhanced_path = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )
        sf.write(enhanced_path.name, audio_data, sample_rate)

        return Path(enhanced_path.name)

    def _enhance_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Apply audio enhancement.

        - Noise reduction via spectral gating
        - Normalization

        Args:
            audio_data: Audio signal
            sample_rate: Sample rate

        Returns:
            Enhanced audio signal
        """
        # Normalize audio
        audio_data = librosa.util.normalize(audio_data)

        # Simple noise reduction using spectral gating
        # Compute short-time Fourier transform
        stft = librosa.stft(audio_data)
        magnitude = np.abs(stft)

        # Estimate noise floor (bottom 10th percentile)
        noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)

        # Apply soft mask (spectral gating)
        mask = magnitude / (magnitude + noise_floor)

        # Apply mask
        stft_cleaned = stft * mask

        # Inverse STFT
        audio_cleaned = librosa.istft(stft_cleaned)

        return audio_cleaned

    def _transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper.

        Args:
            audio_path: Path to audio file
            language: Language code
            task: "transcribe" or "translate"

        Returns:
            Whisper transcription result
        """
        logger.debug(f"Transcribing with Whisper (task={task}, language={language})")

        # Prepare transcription options
        options = {
            "task": task,
            "fp16": self.device == "cuda",
            "verbose": False,
        }

        if language is not None:
            if language not in self.SUPPORTED_LANGUAGES:
                logger.warning(
                    f"Language '{language}' may not be supported. "
                    f"Supported: {self.SUPPORTED_LANGUAGES}"
                )
            options["language"] = language

        # Transcribe
        result = self.whisper_model.transcribe(
            str(audio_path),
            **options
        )

        return result

    def _calculate_confidence(
        self,
        transcription_result: Dict[str, Any]
    ) -> float:
        """
        Calculate average confidence score.

        Whisper provides log probabilities which we convert to confidence.

        Args:
            transcription_result: Whisper result

        Returns:
            Average confidence score (0-1)
        """
        segments = transcription_result.get("segments", [])

        if not segments:
            return 0.0

        # Whisper provides "no_speech_prob" which we invert
        confidences = []
        for segment in segments:
            no_speech_prob = segment.get("no_speech_prob", 0.5)
            confidence = 1.0 - no_speech_prob
            confidences.append(confidence)

        return np.mean(confidences) if confidences else 0.0

    def _diarize(self, audio_path: Path) -> List[Dict[str, Any]]:
        """
        Perform speaker diarization.

        Args:
            audio_path: Path to audio file

        Returns:
            List of speaker segments:
            [
                {
                    "speaker": str,
                    "start": float,
                    "end": float,
                },
                ...
            ]
        """
        if self.diarization_pipeline is None:
            logger.warning("Diarization pipeline not available")
            return []

        logger.debug("Running speaker diarization")

        try:
            # Run diarization
            diarization = self.diarization_pipeline(str(audio_path))

            # Extract speaker segments
            speakers = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end,
                })

            logger.info(f"Identified {len(set(s['speaker'] for s in speakers))} speakers")

            return speakers

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return []

    def _merge_speakers_and_segments(
        self,
        segments: List[Dict[str, Any]],
        speakers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge speaker information with transcription segments.

        Args:
            segments: Whisper segments with timestamps
            speakers: Speaker diarization results

        Returns:
            Segments with speaker labels added
        """
        if not speakers:
            return segments

        # Add speaker to each segment based on temporal overlap
        for segment in segments:
            seg_start = segment["start"]
            seg_end = segment["end"]
            seg_mid = (seg_start + seg_end) / 2

            # Find speaker with maximum overlap
            best_speaker = None
            max_overlap = 0

            for speaker in speakers:
                spk_start = speaker["start"]
                spk_end = speaker["end"]

                # Calculate overlap
                overlap_start = max(seg_start, spk_start)
                overlap_end = min(seg_end, spk_end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = speaker["speaker"]

            segment["speaker"] = best_speaker

        return segments

    def extract_invoice_data(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract invoice data from audio description.

        This is a high-level method that processes audio and attempts
        to extract structured invoice information.

        Args:
            audio_path: Path to audio file
            language: Language code

        Returns:
            {
                "transcription": str,
                "confidence": float,
                "extracted_entities": Dict[str, Any],
                "segments": List[Dict],
            }
        """
        # Process audio
        result = self.process_audio(
            audio_path=audio_path,
            language=language,
            enhance_audio=True
        )

        # Extract entities from transcription
        # In a production system, you would use NER or LLM to extract:
        # - Invoice number
        # - Vendor name
        # - Amounts
        # - Dates
        # - Line items
        # For now, we return the raw transcription

        entities = self._extract_entities_from_text(result["text"])

        return {
            "transcription": result["text"],
            "confidence": result["confidence"],
            "language": result["language"],
            "duration": result["duration"],
            "extracted_entities": entities,
            "segments": result["segments"],
            "speakers": result.get("speakers"),
        }

    def _extract_entities_from_text(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Extract invoice entities from transcription text.

        This is a placeholder for entity extraction logic.
        In production, use NER models or LLM-based extraction.

        Args:
            text: Transcription text

        Returns:
            Extracted entities
        """
        # Placeholder implementation
        # In production, integrate with NER or LLM

        import re

        entities = {
            "invoice_numbers": [],
            "amounts": [],
            "dates": [],
            "vendors": [],
        }

        # Simple regex patterns (improve in production)
        invoice_pattern = r'invoice\s+(?:number\s+)?(\w+[-\w]*)'
        amount_pattern = r'\$?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
        date_pattern = r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b'

        entities["invoice_numbers"] = re.findall(
            invoice_pattern,
            text,
            re.IGNORECASE
        )

        entities["amounts"] = re.findall(amount_pattern, text)
        entities["dates"] = re.findall(date_pattern, text)

        return entities

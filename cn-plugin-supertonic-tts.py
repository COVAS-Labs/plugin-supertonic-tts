
"""
Supertonic TTS Plugin for COVAS:NEXT
Provides offline Text-to-Speech capabilities using the Supertonic model.
"""

from typing import override, Iterable, Any, Optional
import os
import json
import re
import numpy as np
import onnxruntime as ort
import samplerate
from unicodedata import normalize

from lib.PluginHelper import PluginHelper, TTSModel
from lib.PluginSettingDefinitions import (
    PluginSettings,
    ModelProviderDefinition,
    SettingsGrid,
    ParagraphSetting,
    TextSetting,
    NumericalSetting,
)
from lib.PluginBase import PluginBase, PluginManifest
from lib.Logger import log

# --- Helper Classes ---

class UnicodeProcessor:
    def __init__(self, unicode_indexer_path: str):
        with open(unicode_indexer_path, "r") as f:
            self.indexer = json.load(f)

    def _preprocess_text(self, text: str) -> str:
        text = normalize("NFKD", text)
        return text

    def _get_text_mask(self, text_ids_lengths: np.ndarray) -> np.ndarray:
        text_mask = length_to_mask(text_ids_lengths)
        return text_mask

    def _text_to_unicode_values(self, text: str) -> np.ndarray:
        unicode_values = np.array(
            [ord(char) for char in text], dtype=np.uint16
        )  # 2 bytes
        return unicode_values

    def __call__(self, text_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
        text_list = [self._preprocess_text(t) for t in text_list]
        text_ids_lengths = np.array([len(text) for text in text_list], dtype=np.int64)
        text_ids = np.zeros((len(text_list), text_ids_lengths.max()), dtype=np.int64)
        for i, text in enumerate(text_list):
            unicode_vals = self._text_to_unicode_values(text)
            text_ids[i, : len(unicode_vals)] = np.array(
                [self.indexer[val] for val in unicode_vals], dtype=np.int64
            )
        text_mask = self._get_text_mask(text_ids_lengths)
        return text_ids, text_mask

class Style:
    def __init__(self, style_ttl_onnx: np.ndarray, style_dp_onnx: np.ndarray):
        self.ttl = style_ttl_onnx
        self.dp = style_dp_onnx

class TextToSpeech:
    def __init__(self,
        cfgs: dict,
        text_processor: UnicodeProcessor,
        dp_ort: ort.InferenceSession,
        text_enc_ort: ort.InferenceSession,
        vector_est_ort: ort.InferenceSession,
        vocoder_ort: ort.InferenceSession,
    ):
        self.cfgs = cfgs
        self.text_processor = text_processor
        self.dp_ort = dp_ort
        self.text_enc_ort = text_enc_ort
        self.vector_est_ort = vector_est_ort
        self.vocoder_ort = vocoder_ort
        self.sample_rate = cfgs["ae"]["sample_rate"]
        self.base_chunk_size = cfgs["ae"]["base_chunk_size"]
        self.chunk_compress_factor = cfgs["ttl"]["chunk_compress_factor"]
        self.ldim = cfgs["ttl"]["latent_dim"]

    def sample_noisy_latent(
        self, duration: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        bsz = len(duration)
        wav_len_max = duration.max() * self.sample_rate
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = ((wav_len_max + chunk_size - 1) / chunk_size).astype(np.int32)
        latent_dim = self.ldim * self.chunk_compress_factor
        noisy_latent = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)
        latent_mask = get_latent_mask(
            wav_lengths, self.base_chunk_size, self.chunk_compress_factor
        )
        noisy_latent = noisy_latent * latent_mask
        return noisy_latent, latent_mask

    def _infer(
        self, text_list: list[str], style: Style, total_step: int, speed: float = 1.05
    ) -> tuple[np.ndarray, np.ndarray]:
        assert (
            len(text_list) == style.ttl.shape[0]
        ), "Number of texts must match number of style vectors"
        bsz = len(text_list)
        text_ids, text_mask = self.text_processor(text_list)
        dur_onnx, *_ = self.dp_ort.run(
            None, {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask}
        )
        dur_onnx = dur_onnx / speed
        text_emb_onnx, *_ = self.text_enc_ort.run(
            None,
            {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask},
        )
        xt, latent_mask = self.sample_noisy_latent(dur_onnx)
        total_step_np = np.array([total_step] * bsz, dtype=np.float32)
        for step in range(total_step):
            current_step = np.array([step] * bsz, dtype=np.float32)
            xt, *_ = self.vector_est_ort.run(
                None,
                {
                    "noisy_latent": xt,
                    "text_emb": text_emb_onnx,
                    "style_ttl": style.ttl,
                    "text_mask": text_mask,
                    "latent_mask": latent_mask,
                    "current_step": current_step,
                    "total_step": total_step_np,
                },
            )
        wav, *_ = self.vocoder_ort.run(None, {"latent": xt})
        return wav, dur_onnx


def length_to_mask(lengths: np.ndarray, max_len: Optional[int] = None) -> np.ndarray:
    max_len = max_len or lengths.max()
    ids = np.arange(0, max_len)
    mask = (ids < np.expand_dims(lengths, axis=1)).astype(np.float32)
    return mask.reshape(-1, 1, max_len)

def get_latent_mask(
    wav_lengths: np.ndarray, base_chunk_size: int, chunk_compress_factor: int
) -> np.ndarray:
    latent_size = base_chunk_size * chunk_compress_factor
    latent_lengths = (wav_lengths + latent_size - 1) // latent_size
    latent_mask = length_to_mask(latent_lengths)
    return latent_mask

def chunk_text(text: str, max_len: int = 300) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    chunks = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        pattern = r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+"
        sentences = re.split(pattern, paragraph)
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_len:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
    return chunks

def load_voice_style(voice_style_path: str) -> Style:
    with open(voice_style_path, "r") as f:
        voice_style = json.load(f)
    ttl_dims = voice_style["style_ttl"]["dims"]
    dp_dims = voice_style["style_dp"]["dims"]
    ttl_data = np.array(voice_style["style_ttl"]["data"], dtype=np.float32).flatten()
    ttl_style = ttl_data.reshape(1, ttl_dims[1], ttl_dims[2])
    dp_data = np.array(voice_style["style_dp"]["data"], dtype=np.float32).flatten()
    dp_style = dp_data.reshape(1, dp_dims[1], dp_dims[2])
    return Style(ttl_style, dp_style)

# --- Plugin Implementation ---

class SupertonicTTSModel(TTSModel):
    """Supertonic Text-to-Speech model implementation."""
    
    def __init__(self, model_dir: str, voice: str = "M1", speed: float = 1.0):
        super().__init__("supertonic-tts")
        self.model_dir = model_dir
        self.voice = voice
        self.speed = speed
        self._tts_engine: Optional[TextToSpeech] = None
        self._voices: dict[str, Style] = {}
        
    def _load_models(self):
        if self._tts_engine is not None:
            return

        log('info', f"Loading Supertonic models from {self.model_dir}")
        
        try:
            paths = {
                "duration_predictor": os.path.join(self.model_dir, "duration_predictor.onnx"),
                "text_encoder": os.path.join(self.model_dir, "text_encoder.onnx"),
                "vector_estimator": os.path.join(self.model_dir, "vector_estimator.onnx"),
                "vocoder": os.path.join(self.model_dir, "vocoder.onnx"),
                "config": os.path.join(self.model_dir, "tts.json"),
                "indexer": os.path.join(self.model_dir, "unicode_indexer.json")
            }
            
            for name, path in paths.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Model file not found: {path}")

            with open(paths["config"], "r") as f:
                cfgs = json.load(f)
                
            text_processor = UnicodeProcessor(paths["indexer"])
            
            opts = ort.SessionOptions()
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                log('info', "Supertonic: Using GPU for inference")
            else:
                log('info', "Supertonic: Using CPU for inference")

            dp_ort = ort.InferenceSession(paths["duration_predictor"], sess_options=opts, providers=providers)
            text_enc_ort = ort.InferenceSession(paths["text_encoder"], sess_options=opts, providers=providers)
            vector_est_ort = ort.InferenceSession(paths["vector_estimator"], sess_options=opts, providers=providers)
            vocoder_ort = ort.InferenceSession(paths["vocoder"], sess_options=opts, providers=providers)
            
            self._tts_engine = TextToSpeech(
                cfgs, text_processor, dp_ort, text_enc_ort, vector_est_ort, vocoder_ort
            )
            
        except Exception as e:
            log('error', f"Failed to load Supertonic models: {e}")
            raise

    def _get_voice_style(self, voice_name: str) -> Style:
        if voice_name in self._voices:
            return self._voices[voice_name]
            
        voice_path = os.path.join(self.model_dir, "voices", f"{voice_name}.json")
        if not os.path.exists(voice_path):
            log('warning', f"Voice {voice_name} not found at {voice_path}, trying M1")
            voice_path = os.path.join(self.model_dir, "voices", "M1.json")
            if not os.path.exists(voice_path):
                 raise FileNotFoundError(f"Voice file not found: {voice_path}")
        
        style = load_voice_style(voice_path)
        self._voices[voice_name] = style
        return style

    @override
    def synthesize(self, text: str, voice: str) -> Iterable[bytes]:
        self._load_models()
        if self._tts_engine is None:
             raise RuntimeError("Supertonic TTS engine not initialized")

        target_voice = voice if voice else self.voice
        if target_voice == 'nova':
            target_voice = 'F1'  # Map 'nova' to 'F1' voice for OpenAI compatibility
        style = self._get_voice_style(target_voice)
        
        text_list = chunk_text(text)
        total_step = 5
        silence_duration = 0.3
        
        for i, chunk in enumerate(text_list):
            wav, dur_onnx = self._tts_engine._infer([chunk], style, total_step, self.speed)
            
            samples = wav.flatten()
            sample_rate = self._tts_engine.sample_rate
            
            if sample_rate != 24000:
                samples = samplerate.resample(samples, 24000 / sample_rate, "sinc_best")
            
            samples_int16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)
            
            # Yield in 100ms chunks (2400 samples at 24kHz)
            chunk_size = 2400
            for j in range(0, len(samples_int16), chunk_size):
                yield samples_int16[j:j + chunk_size].tobytes()
            
            if i < len(text_list) - 1:
                silence_samples = int(silence_duration * 24000)
                silence = np.zeros(silence_samples, dtype=np.int16)
                for j in range(0, len(silence), chunk_size):
                    yield silence[j:j + chunk_size].tobytes()

class SupertonicPlugin(PluginBase):
    """
    Plugin providing Supertonic Text-to-Speech services.
    """
    
    def __init__(self, plugin_manifest: PluginManifest):
        super().__init__(plugin_manifest)
        
        self.settings = PluginSettings(
            key="Supertonic TTS",
            label="Supertonic TTS",
            icon="volume-high",
            grids=[
                SettingsGrid(
                    key="general",
                    label="General",
                    fields=[
                        ParagraphSetting(
                            key="info_text",
                            label=None,
                            type="paragraph",
                            readonly=False,
                            placeholder=None,
                            
                            content="To use Supertonic TTS, select it as your *TTS provider* in *Advanced → TTS Settings*."
                        ),
                    ]
                ),
            ]
        )
        
        
        self.model_providers = [
            ModelProviderDefinition(
                kind='tts',
                id='supertonic-tts',
                label='Supertonic TTS (Offline)',
                settings_config=[
                    SettingsGrid(
                        key='settings',
                        label='Settings',
                        fields=[
                            TextSetting(
                                key='voice',
                                label='Voice (M1, M2, F1, F2)',
                                type='text',
                                readonly=False,
                                placeholder='M1',
                                default_value='M1',
                                max_length=None,
                                min_length=None,
                                hidden=False,
                            ),
                            NumericalSetting(
                                key='speed',
                                label='Speed',
                                type='number',
                                readonly=False,
                                placeholder='1.0',
                                default_value=1.0,
                                min_value=0.5,
                                max_value=2.0,
                                step=0.1,
                            )
                        ]
                    )
                ]
            )
        ]

    @override
    def create_model(self, provider_id: str, settings: dict[str, Any]) -> TTSModel:
        """Create a model instance for the given provider."""
        
        if provider_id == 'supertonic-tts':
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(plugin_dir, "model")
            
            voice = settings.get('voice', 'M1')
            speed = float(settings.get('speed', 1.0))
            
            return SupertonicTTSModel(model_dir=model_dir, voice=voice, speed=speed)
            
        raise ValueError(f'Unknown Supertonic provider: {provider_id}')

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "deps")
    plugin_manifest = PluginManifest(
        name="Supertonic TTS Plugin",
        version="1.0.0",
        author="COVAS:NEXT",
        description="Supertonic TTS Plugin for COVAS:NEXT"
    )
    plugin = SupertonicPlugin(plugin_manifest)
    try:
        model = plugin.create_model('supertonic-tts', {})
        log('info', "Supertonic TTS Plugin initialized successfully.")
    except Exception as e:
        log('error', f"Failed to initialize plugin: {e}")



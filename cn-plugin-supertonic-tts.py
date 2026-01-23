
"""
Supertonic TTS Plugin for COVAS:NEXT
Provides offline Text-to-Speech capabilities using the Supertonic model.
"""

from typing import override, Iterable, Any, Optional
import os
import re
import numpy as np
import samplerate

# Use upstream (MIT-licensed) Supertonic 2 helper implementation, vendored into this plugin.
from .vendor.supertone_supertonic_py.helper import (
    AVAILABLE_LANGS,
    TextToSpeech,
    Style,
    chunk_text,
    load_text_to_speech,
    load_voice_style,
)

from lib.PluginHelper import PluginHelper, TTSModel
from lib.PluginSettingDefinitions import (
    PluginSettings,
    ModelProviderDefinition,
    SettingsGrid,
    ParagraphSetting,
    SelectSetting,
    NumericalSetting,
)
from lib.PluginBase import PluginBase, PluginManifest
from lib.Logger import log

"""Plugin implementation."""

# --- Plugin Implementation ---

class SupertonicTTSModel(TTSModel):
    """Supertonic Text-to-Speech model implementation."""
    
    def __init__(
        self,
        model_dir: str,
        voice: str = "M1",
        speed: float = 1.05,
        language: str = "en",
    ):
        super().__init__("supertonic-tts")
        self.model_dir = model_dir
        self.voice = voice
        self.speed = speed
        self.language = language
        self._tts_engine: Optional[TextToSpeech] = None
        self._voices: dict[str, Style] = {}
        self._onnx_dir: Optional[str] = None
        self._voice_dir: Optional[str] = None
        
    def _load_models(self):
        if self._tts_engine is not None:
            return

        log('info', f"Loading Supertonic models from {self.model_dir}")
        
        try:
            onnx_dir_candidates = [os.path.join(self.model_dir, "onnx"), self.model_dir]
            onnx_dir = None
            for candidate in onnx_dir_candidates:
                required = [
                    os.path.join(candidate, "duration_predictor.onnx"),
                    os.path.join(candidate, "text_encoder.onnx"),
                    os.path.join(candidate, "vector_estimator.onnx"),
                    os.path.join(candidate, "vocoder.onnx"),
                    os.path.join(candidate, "tts.json"),
                    os.path.join(candidate, "unicode_indexer.json"),
                ]
                if all(os.path.exists(p) for p in required):
                    onnx_dir = candidate
                    break
            if onnx_dir is None:
                raise FileNotFoundError(
                    f"Could not find a valid Supertonic ONNX directory in {onnx_dir_candidates}. "
                    "Expected duration_predictor.onnx, text_encoder.onnx, vector_estimator.onnx, vocoder.onnx, tts.json, unicode_indexer.json"
                )

            voice_dir_candidates = [
                os.path.join(self.model_dir, "voice_styles"),
                os.path.join(self.model_dir, "voices"),
            ]
            voice_dir = next((d for d in voice_dir_candidates if os.path.isdir(d)), None)
            if voice_dir is None:
                raise FileNotFoundError(
                    f"Could not find voice styles directory in {voice_dir_candidates}. "
                    "Expected model/voice_styles or model/voices"
                )

            self._onnx_dir = onnx_dir
            self._voice_dir = voice_dir

            # Upstream helper handles model/config/indexer/session loading.
            # GPU mode is intentionally disabled (upstream marks it as not fully tested).
            log('info', "Supertonic: Using CPU for inference")
            self._tts_engine = load_text_to_speech(onnx_dir, use_gpu=False)
            
        except Exception as e:
            log('error', f"Failed to load Supertonic models: {e}")
            raise

    def _get_voice_style(self, voice_name: str) -> Style:
        voice_dir = self._voice_dir or os.path.join(self.model_dir, "voices")

        # COVAS may pass through provider-agnostic voice names.
        # Map common aliases to Supertonic voice presets when possible.
        alias_map = {
            "nova": "F1",
            "alloy": "M1",
            "echo": "M2",
            # Map other common OpenAI-style names to the closest available presets.
            "fable": "M1",
            "onyx": "M2",
            "shimmer": "F2",
        }

        requested = (voice_name or "").strip()
        mapped = alias_map.get(requested.lower(), requested)

        def _voice_path_for(name: str) -> str:
            fname = name if name.lower().endswith(".json") else f"{name}.json"
            return os.path.join(voice_dir, fname)

        # Resolve to an existing JSON file.
        candidates: list[str] = []
        if mapped:
            candidates.append(mapped)
        if requested and requested != mapped:
            candidates.append(requested)

        configured_default = (self.voice or "").strip() or "M1"
        if configured_default not in candidates:
            candidates.append(configured_default)
        if "M1" not in candidates:
            candidates.append("M1")

        resolved_name = None
        resolved_path = None
        for candidate in candidates:
            candidate_path = _voice_path_for(candidate)
            if os.path.exists(candidate_path):
                resolved_name = candidate
                resolved_path = candidate_path
                break

        if resolved_name is None or resolved_path is None:
            raise FileNotFoundError(
                f"No voice style JSON found. Tried: {', '.join(_voice_path_for(c) for c in candidates)}"
            )

        if resolved_name in self._voices:
            return self._voices[resolved_name]

        if requested and resolved_name.lower() != requested.lower():
            log('info', f"Using voice '{resolved_name}' (requested '{requested}')")

        style = load_voice_style([resolved_path], verbose=False)
        self._voices[resolved_name] = style
        return style

    def _normalize_numbers(self, text: str, lang: str) -> str:
        """Convert standalone numeric tokens to localized words when possible.

        Supertonic's character-level model can produce unstable output on raw digits
        in some cases; converting to words improves pronunciation stability.
        """

        try:
            from num2words import num2words  # type: ignore
        except Exception:
            return text

        lang = (lang or "en").strip().lower()
        lang_map = {
            "en": "en",
            "ko": "ko",
            "es": "es",
            "pt": "pt",
            "fr": "fr",
        }
        target_lang = lang_map.get(lang, "en")

        decimal_comma_langs = {"fr", "es", "pt"}

        # Match standalone numeric tokens (no adjacent letters/underscore).
        number_pattern = re.compile(r"(?<![\w])([+-]?\d+(?:[\.,]\d+)*)(?![\w])")

        def _parse_number_token(token: str) -> int | float | None:
            raw = token.replace(" ", "")

            if lang in decimal_comma_langs:
                # Treat ',' as decimal separator; drop '.' thousand separators.
                raw = raw.replace(".", "")
                raw = raw.replace(",", ".")
            else:
                # Treat '.' as decimal separator; drop ',' thousand separators.
                raw = raw.replace(",", "")

            try:
                if "." in raw:
                    return float(raw)
                return int(raw)
            except Exception:
                return None

        def _replace(match: re.Match[str]) -> str:
            token = match.group(1)

            # Skip common date/time patterns like 2026-01-23 or 3:45.
            end = match.end(1)
            start = match.start(1)
            if end < len(text) and text[end] in "-/:" and end + 1 < len(text) and text[end + 1].isdigit():
                return token
            if start > 0 and text[start - 1] in "-/:" and start - 2 >= 0 and text[start - 2].isdigit():
                return token

            value = _parse_number_token(token)
            if value is None:
                return token

            try:
                spoken = num2words(value, lang=target_lang)
            except Exception:
                return token

            spoken = re.sub(r"\s+", " ", str(spoken)).strip()
            return spoken

        return number_pattern.sub(_replace, text)

    @override
    def synthesize(self, text: str, voice: str) -> Iterable[bytes]:
        self._load_models()
        if self._tts_engine is None:
             raise RuntimeError("Supertonic TTS engine not initialized")

        lang = (self.language or "en").strip().lower()
        if lang not in AVAILABLE_LANGS:
            log('warning', f"Unsupported language '{lang}', falling back to 'en'")
            lang = "en"

        # Prefer explicitly requested voice if it resolves, otherwise fall back to configured default.
        target_voice = voice if voice else self.voice
        style = self._get_voice_style(target_voice)
        
        max_len = 120 if lang == "ko" else 300
        text = self._normalize_numbers(text, lang)
        text_list = chunk_text(text, max_len=max_len)
        total_step = 5
        silence_duration = 0.3
        
        for i, chunk in enumerate(text_list):
            wav, dur_onnx = self._tts_engine._infer([chunk], [lang], style, total_step, self.speed)

            # Trim vocoder output to predicted duration (matches upstream examples)
            try:
                dur_seconds = float(np.asarray(dur_onnx).reshape(-1)[0])
            except Exception:
                dur_seconds = float(dur_onnx)
            wav_len = int(self._tts_engine.sample_rate * max(dur_seconds, 0.0))
            wav_2d = np.asarray(wav)
            if wav_2d.ndim == 2:
                wav_1d = wav_2d[0, : min(wav_len, wav_2d.shape[1])]
            else:
                wav_1d = wav_2d.reshape(-1)[:wav_len]

            samples = wav_1d
            sample_rate = self._tts_engine.sample_rate
            
            if sample_rate != 24000:
                if samplerate is None:
                    raise RuntimeError(
                        "Resampling required (model sample_rate != 24000) but 'samplerate' is not available. "
                        "Install dependencies (or use a 24kHz model) to proceed."
                    )
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

        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(plugin_dir, "model")
        voice_select_options = self._build_voice_select_options(model_dir)
        language_select_options = self._build_language_select_options()
        
        self.settings_config = PluginSettings(
            key="Supertonic TTS",
            label="Supertonic TTS",
            icon="volume_up",
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
                            
                            content='To use Supertonic TTS, select it as your "TTS provider" in "Advanced" → "TTS Settings".'
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
                            SelectSetting(
                                key='voice',
                                label='Voice style',
                                type='select',
                                readonly=False,
                                placeholder=None,
                                default_value='M1',
                                select_options=voice_select_options,
                                multi_select=False,
                            ),
                            SelectSetting(
                                key='language',
                                label='Language',
                                type='select',
                                readonly=False,
                                placeholder=None,
                                default_value='en',
                                select_options=language_select_options,
                                multi_select=False,
                            ),
                            NumericalSetting(
                                key='speed',
                                label='Speed',
                                type='number',
                                readonly=False,
                                placeholder='1.05',
                                default_value=1.05,
                                min_value=0.8,
                                max_value=1.8,
                                step=0.1,
                            )
                        ]
                    )
                ]
            )
        ]

    def _build_language_select_options(self) -> list[dict[str, object]]:
        # SelectOption: { key, label, value, disabled }
        labels = {
            "en": "English (en)",
            "ko": "Korean (ko)",
            "es": "Spanish (es)",
            "pt": "Portuguese (pt)",
            "fr": "French (fr)",
        }
        options: list[dict[str, object]] = []
        for lang in AVAILABLE_LANGS:
            options.append(
                {
                    "key": lang,
                    "label": labels.get(lang, lang),
                    "value": lang,
                    "disabled": False,
                }
            )
        return options

    def _build_voice_select_options(self, model_dir: str) -> list[dict[str, object]]:
        # Voice files are shipped as JSON in model/voices (or model/voice_styles).
        voice_dir_candidates = [
            os.path.join(model_dir, "voices"),
            os.path.join(model_dir, "voice_styles"),
        ]
        voice_dir = next((d for d in voice_dir_candidates if os.path.isdir(d)), None)
        if voice_dir is None:
            raise FileNotFoundError(
                f"Could not find voice styles directory in {voice_dir_candidates}. "
                "Expected model/voice_styles or model/voices"
            )

        names: list[str] = []
        for fname in os.listdir(voice_dir):
            if not fname.lower().endswith(".json"):
                continue
            if fname.startswith("."):
                continue
            names.append(os.path.splitext(fname)[0])

        def _sort_key(name: str) -> tuple[int, int, str]:
            m = re.match(r"^([MmFf])(\d+)$", name)
            if m:
                gender = m.group(1).upper()
                num = int(m.group(2))
                gender_rank = 0 if gender == "M" else 1
                return (0, gender_rank * 100 + num, name)
            return (1, 9999, name)

        names = sorted(set(names), key=_sort_key)
        if not names:
            names = ["M1"]

        return [{"key": v, "label": v, "value": v, "disabled": False} for v in names]

    @override
    def create_model(self, provider_id: str, settings: dict[str, Any]) -> TTSModel:
        """Create a model instance for the given provider."""
        
        if provider_id == 'supertonic-tts':
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(plugin_dir, "model")
            
            voice = settings.get('voice', 'M1')
            language = settings.get('language', 'en')
            speed = float(settings.get('speed', 1.0))
            
            return SupertonicTTSModel(model_dir=model_dir, voice=voice, speed=speed, language=language)
            
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



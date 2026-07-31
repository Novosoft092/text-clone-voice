from cog import BasePredictor, Input, Path
import os
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import soundfile as sf
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from pydub import AudioSegment
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)
from f5_tts.model import DiT


class Predictor(BasePredictor):
    def setup(self):
        print("[setup] Starting model load...", flush=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=self.device)
        self.ema_model = load_model(
            DiT,
            dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            mel_spec_type="vocos",
            vocab_file=hf_hub_download("ai4bharat/IndicF5", filename="checkpoints/vocab.txt"),
            device=self.device,
        )
        ckpt_path = hf_hub_download("ai4bharat/IndicF5", filename="model.safetensors")
        print(f"[setup] Loading checkpoint from {ckpt_path}", flush=True)
        raw_sd = load_file(ckpt_path, device=self.device)
        print(f"[setup] sample keys: {list(raw_sd.keys())[:3]}", flush=True)
        sd = {}
        for k in raw_sd:
            nk = k[10:] if k.startswith("ema_model.") else k
            sd[nk] = raw_sd[k]
        missing, unexpected = self.ema_model.load_state_dict(sd, strict=False)
        print(f"[setup] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        print("[setup] Model loaded successfully.", flush=True)

    def predict(self, text: str = Input(description="Text to speak"), ref_audio: Path = Input(description="Reference speaker audio wav"), ref_text: str = Input(description="Reference audio transcript")) -> Path:
        print("[predict] Running inference...", flush=True)
        ref_audio_path = str(ref_audio)
        data, sr = sf.read(ref_audio_path)
        max_seconds = 12
        max_samples = int(max_seconds * sr)
        if len(data) > max_samples:
            print("[predict] Trimming reference audio", flush=True)
            data = data[:max_samples]
            ref_audio_path = "/tmp/ref_trimmed.wav"
            sf.write(ref_audio_path, data, sr)

        ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(ref_audio_path, ref_text)

        audio, final_sample_rate, _ = infer_process(
            ref_audio_proc,
            ref_text_proc,
            text,
            self.ema_model,
            self.vocoder,
            mel_spec_type="vocos",
            speed=1.0,
            device=self.device,
        )

        audio = np.asarray(audio, dtype=np.float32)
        print(f"[predict] Raw audio stats: min={audio.min():.4f} max={audio.max():.4f} std={audio.std():.4f}", flush=True)

        audio_int16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=final_sample_rate,
            sample_width=2,
            channels=1,
        )

        target_dBFS = -20.0
        if audio_segment.dBFS != float("-inf"):
            change_in_dBFS = target_dBFS - audio_segment.dBFS
            audio_segment = audio_segment.apply_gain(change_in_dBFS)

        final_audio = np.array(audio_segment.get_array_of_samples())

        out = "/tmp/out.wav"
        sf.write(out, final_audio, samplerate=final_sample_rate, subtype="PCM_16")
        print("[predict] Inference complete.", flush=True)
        return Path(out)

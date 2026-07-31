from cog import BasePredictor, Input, Path
import os
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import soundfile as sf
import numpy as np
from transformers import AutoModel

class Predictor(BasePredictor):
      def setup(self):
            print("[setup] Starting model load...", flush=True)
            self.model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True, remove_sil=False)
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
            wav = self.model(text, ref_audio_path=ref_audio_path, ref_text=ref_text)
            audio = np.array(wav)
            if audio.dtype == np.int16:
                  audio = audio.astype(np.float32) / 32768.0
            out = "/tmp/out.wav"
            sf.write(out, np.array(audio, dtype=np.float32), samplerate=24000)
            print("[predict] Inference complete.", flush=True)
            return Path(out)
                  

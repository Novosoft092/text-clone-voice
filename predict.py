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
                self.model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
                print("[setup] Model loaded successfully.", flush=True)

      def predict(self, text: str = Input(description="Text to speak"), ref_audio: Path = Input(description="Reference speaker audio wav"), ref_text: str = Input(description="Reference audio transcript")) -> Path:
                print("[predict] Running inference...", flush=True)
                wav = self.model(text, ref_audio_path=str(ref_audio), ref_text=ref_text)
                out = "/tmp/out.wav"
                sf.write(out, np.array(wav, dtype=np.float32), samplerate=24000)
                print("[predict] Inference complete.", flush=True)
                return Path(out)
        

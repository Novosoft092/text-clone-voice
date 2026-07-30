from cog import BasePredictor, Input, Path
import soundfile as sf
import numpy as np
from transformers import AutoModel

class Predictor(BasePredictor):
  def setup(self):
    self.model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)

def predict(self, text: str = Input(description="Text to speak"), ref_audio: Path = Input(description="Reference speaker audio wav"), ref_text: str = Input(description="Reference audio transcript")) -> Path:
  wav = self.model(text, ref_audio_path=str(ref_audio), ref_text=ref_text)
  out = "/tmp/out.wav"
  sf.write(out, np.array(wav, dtype=np.float32), samplerate=24000)
  return Path(out)

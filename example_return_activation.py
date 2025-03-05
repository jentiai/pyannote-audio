import os

import numpy as np
import torch

from pyannote.audio import Pipeline

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN", None)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-2.1",  # or 3.1
    use_auth_token=HF_AUTH_TOKEN)

# send pipeline to GPU (when available)
pipeline.to(torch.device("cuda"))

# apply pretrained pipeline
diarization, activations = pipeline("./tests/data/trn07.wav", return_activations=True)

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

# print activations

print(activations)
print(f"shape={activations.data.shape}")
print(f"duration={activations.sliding_window.duration:.1f}s")
print(f"step={activations.sliding_window.step:.1f}s")
# show all numpy array
# 배열 전체가 출력되도록 threshold를 무한대로 설정
np.set_printoptions(threshold=np.inf)
print(f"speaker 0 = {activations.data[:, 0]}")

"""Private GPU service. Only media-relative paths cross the container boundary."""
import os
import threading
import time
import warnings
import gc
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(os.getenv('STORAGE_ROOT', '/data')).resolve()
MODEL_DIR = Path(os.getenv('MODEL_DIR', '/models/checkpoints'))
STATE = {'status': 'loading', 'detail': 'Downloading IndexTTS2 weights', 'model': 'IndexTTS2 2.0.0'}
LOCK = threading.Lock()
MODEL = None
MAX_TOKENS = int(os.getenv('TTS_MAX_TEXT_TOKENS', '120'))


def media_path(key: str) -> Path:
    path = (ROOT / key).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise HTTPException(400, 'Invalid media key')
    return path


def load_model():
    global MODEL
    try:
        from huggingface_hub import snapshot_download
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download('IndexTeam/IndexTTS-2', revision='740dcaff396282ffb241903d150ac011cd4b1ede',
                          local_dir=str(MODEL_DIR), max_workers=2,
                          allow_patterns=['*.pth', '*.pt', '*.model', '*.yaml', '*.json', 'qwen0.6bemo4-merge/*'])
        STATE['detail'] = 'Loading models onto GPU'
        # Upstream sets HF_HUB_CACHE relative to its cwd; put that cache on our persistent volume.
        os.chdir('/models')
        from indextts.infer_v2 import IndexTTS2
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable inside the TTS container')
        MODEL = IndexTTS2(cfg_path=str(MODEL_DIR / 'config.yaml'), model_dir=str(MODEL_DIR),
                          device='cuda:0', use_fp16=True, use_cuda_kernel=False, use_deepspeed=False)
        # This product uses original-segment audio emotion prompts, never Qwen text emotion.
        MODEL.qwen_emo.model.to('cpu')
        torch.cuda.empty_cache()
        STATE.update(status='ready', detail=torch.cuda.get_device_name(0))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        STATE.update(status='error', detail=f'{type(exc).__name__}: {exc}')


@asynccontextmanager
async def lifespan(app):
    def warmup():
        with LOCK:load_model()
    threading.Thread(target=warmup, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


def unload_model():
    global MODEL
    MODEL = None
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    STATE.update(status='ready',detail='GPU 就绪；IndexTTS2 按需加载')


def ensure_model():
    if MODEL is None:
        load_model()
    if MODEL is None:
        raise HTTPException(503,STATE['detail'])


from demucs_service import install
install(app, LOCK, media_path, unload_model)


@app.get('/health')
def health():
    if STATE['status'] != 'ready':
        from fastapi.responses import JSONResponse
        return JSONResponse(STATE, status_code=503)
    return {**STATE, 'max_text_tokens_per_segment': MAX_TOKENS, 'max_text_characters': 5000,
            'max_reference_seconds': 15, 'languages': ['en', 'zh']}


class Texts(BaseModel):
    texts: list[str] = Field(max_length=500)


@app.post('/tokenize')
def tokenize(body: Texts):
    with LOCK:
        ensure_model()
        return {'counts': [len(MODEL.tokenizer.tokenize(t)) for t in body.texts],
                'chunk_limit': MAX_TOKENS}


class Speech(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    speaker_audio: str
    emotion_audio: str
    output: str


@app.post('/synthesize')
def synthesize(body: Speech):
    import soundfile as sf
    import numpy as np
    speaker, emotion, output = (media_path(k) for k in (body.speaker_audio, body.emotion_audio, body.output))
    if output.suffix != '.wav':
        raise HTTPException(400, 'Output must be WAV')
    for kind, path in [('speaker', speaker), ('emotion', emotion)]:
        if not path.is_file():
            raise HTTPException(422, f'{kind} reference is missing')
        signal, sr = sf.read(path)
        duration = len(signal) / sr
        if duration <= 0 or not np.isfinite(signal).all():
            raise HTTPException(422, f'Unsuitable {kind} reference: duration or audio quality')
    with LOCK:
        ensure_model()
        count = len(MODEL.tokenizer.tokenize(body.text))
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.partial.wav')
        started = time.monotonic()
        try:
            import torch
            torch.manual_seed(42)
            with warnings.catch_warnings():
                # Upstream otherwise saves incomplete speech when its generation cap is hit.
                warnings.filterwarnings('error', message='WARN: generation stopped.*', category=RuntimeWarning)
                for chunk_tokens in dict.fromkeys([MAX_TOKENS, max(20, MAX_TOKENS//2)]):
                    try:
                        MODEL.infer(spk_audio_prompt=str(speaker), emo_audio_prompt=str(emotion), text=body.text,
                                    output_path=str(temporary), emo_alpha=1.0, use_random=False,
                                    max_text_tokens_per_segment=chunk_tokens, num_beams=1, verbose=False)
                        break
                    except RuntimeWarning:
                        temporary.unlink(missing_ok=True)
                        if chunk_tokens == max(20, MAX_TOKENS//2):
                            raise
            signal, sr = sf.read(temporary)
            if not len(signal) or not np.isfinite(signal).all() or np.sqrt(np.mean(signal**2)) < 0.0001:
                raise RuntimeError('Synthesized audio is empty or invalid')
            temporary.replace(output)
            return {'output': body.output, 'duration': len(signal) / sr, 'tokens': count,
                    'elapsed_seconds': round(time.monotonic() - started, 2)}
        except RuntimeWarning as exc:
            temporary.unlink(missing_ok=True)
            raise HTTPException(422, 'Incomplete synthesis: shorten this utterance at a natural boundary') from exc
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            import torch
            torch.cuda.empty_cache()
            if 'out of memory' in str(exc).lower():
                raise HTTPException(503, 'GPU memory exhausted; retry after freeing GPU memory or use a larger GPU') from exc
            raise


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)

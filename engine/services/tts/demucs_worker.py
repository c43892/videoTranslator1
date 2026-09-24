"""Bounded-memory GPU separation process. Exiting releases its CUDA context."""
import json
import os
from pathlib import Path
import subprocess
import sys


def run(config):
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable for Demucs')
    model = get_model(config['model'])
    rate = model.samplerate
    out = Path(config['directory'])
    converted = out/'source-44100.wav'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',config['input'],
        '-vn','-ar',str(rate),'-ac','2','-c:a','pcm_f32le',str(converted)],check=True)
    paths = {name:out/(name+'.partial.wav') for name in ('dialogue','background','effects')}
    with sf.SoundFile(converted) as source, \
         sf.SoundFile(paths['dialogue'],'w',samplerate=rate,channels=2,subtype='FLOAT') as voice, \
         sf.SoundFile(paths['background'],'w',samplerate=rate,channels=2,subtype='FLOAT') as background, \
         sf.SoundFile(paths['effects'],'w',samplerate=rate,channels=2,subtype='FLOAT') as effects:
        step, overlap = 60*rate, 2*rate
        total, start, pending = len(source), 0, None
        while start < total:
            end = min(total,start+step+overlap)
            source.seek(start)
            audio = source.read(end-start,dtype='float32',always_2d=True)
            wav = torch.from_numpy(audio.T.copy())
            ref = wav.mean(0)
            mean, std = ref.mean(), ref.std()
            if float(std) < 1e-7:
                vocals = np.zeros_like(audio)
            else:
                with torch.inference_mode():
                    estimates = apply_model(model,((wav-mean)/std)[None],device='cuda',
                        shifts=0,split=True,segment=config['segment'],overlap=.25,num_workers=0)[0]
                vocals = (estimates[model.sources.index('vocals')]*std+mean).T.cpu().numpy().copy()
                del estimates
            if pending is not None:
                fade = np.linspace(0,1,len(pending),dtype='float32')[:,None]
                vocals[:len(pending)] = pending*(1-fade)+vocals[:len(pending)]*fade
            final = end == total
            count = len(audio) if final else step
            if not np.isfinite(vocals).all():
                raise RuntimeError('Demucs produced non-finite audio')
            voice.write(vocals[:count])
            background.write(audio[:count]-vocals[:count])
            effects.write(np.zeros((count,2),dtype='float32'))
            pending = vocals[count:].copy() if not final else None
            progress = out/'progress.json'
            progress.with_suffix('.tmp').write_text(json.dumps({'progress':round(100*(start+count)/total)}))
            progress.with_suffix('.tmp').replace(progress)
            if final:break
            start += step
    for name,path in paths.items():
        if sf.info(path).frames != total:
            raise RuntimeError('Demucs output length mismatch')
        path.replace(out/(name+'.wav'))
    converted.unlink(missing_ok=True)
    (out/'result.json').write_text(json.dumps({'frames':total,'sample_rate':rate,
        'model':config['model'],'version':'4.0.1','layout':'vocals + mixture-minus-vocals',
        'peak_gpu_mb':round(torch.cuda.max_memory_allocated()/1024**2)}))


if __name__ == '__main__':
    run(json.loads(Path(sys.argv[1]).read_text()))

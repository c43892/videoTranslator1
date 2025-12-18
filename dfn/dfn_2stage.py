import argparse
import glob
import os
import shutil
import subprocess
import time
import librosa
from typing import Tuple, List, Callable, Optional

import numpy as np
import torch
import torchaudio as ta
import soundfile as sf


def _safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def _rm_tree_contents(d: str):
    """删除目录内容（不删目录本身）"""
    if not os.path.isdir(d):
        return
    for p in glob.glob(os.path.join(d, "*")):
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
        except OSError:
            pass


def _wait_file_stable(path: str, timeout_s: float = 120.0) -> None:
    """等待文件出现且大小稳定（Windows 写入/锁常见）"""
    t0 = time.time()
    last = -1
    stable = 0
    while time.time() - t0 < timeout_s:
        if os.path.exists(path):
            try:
                sz = os.path.getsize(path)
                if sz > 0 and sz == last:
                    stable += 1
                    if stable >= 3:
                        return
                else:
                    stable = 0
                    last = sz
            except OSError:
                pass
        time.sleep(0.3)
    raise RuntimeError(f"File not ready/stable: {path}")


def _wait_any_wav_created(outdir: str, t_start: float, timeout_s: float = 120.0) -> str:
    """等待 outdir 出现新 wav（mtime >= t_start）并且稳定"""
    t0 = time.time()
    last_seen = None
    while time.time() - t0 < timeout_s:
        cands = []
        for p in glob.glob(os.path.join(outdir, "*.wav")):
            try:
                mtime = os.path.getmtime(p)
                sz = os.path.getsize(p)
                if mtime >= t_start and sz > 0:
                    cands.append((mtime, p))
            except OSError:
                pass

        if cands:
            cands.sort()
            last_seen = cands[-1][1]
            _wait_file_stable(last_seen, timeout_s=timeout_s)
            return last_seen

        time.sleep(0.2)
    raise RuntimeError(f"deepFilter did not create wav in: {outdir}. Last seen: {last_seen}")


def _get_audio_info(path: str):
    info = ta.info(path)
    return info.sample_rate, info.num_frames


def stream_process(
    inputs: List[str],
    output: str,
    process_fn: Optional[Callable[[List[np.ndarray], int], np.ndarray]] = None,
    chunk_dur_s: float = 30.0,
    target_sr: Optional[int] = None
):
    """
    流式处理多个输入音频，防止 OOM。
    inputs: 输入文件路径列表
    output: 输出文件路径
    process_fn: 处理函数，接收 (input_chunks, sample_rate) 返回 processed_chunk (numpy [N, C])
                如果为 None，则直接把 inputs[0] 的 chunk 写出（相当于复制/拼接）
    chunk_dur_s: 每次处理的时长
    target_sr: 强制输出采样率，默认为 inputs[0] 的采样率
    """
    if not inputs:
        raise ValueError("No input files")

    # 获取基础信息
    sr0, frames0 = _get_audio_info(inputs[0])
    if target_sr is None:
        target_sr = sr0
    
    # 检查所有输入文件的采样率是否一致（建议一致，否则这里不处理重采样会很麻烦，
    # 实际场景 deepFilter 输出应该是一致的）
    for p in inputs[1:]:
        sr, _ = _get_audio_info(p)
        if sr != sr0:
             # 简单的 check，实际工程可能需要 on-the-fly resample，但这里假设流程控制正常
            print(f"Warning: Sample rate mismatch {p} ({sr}) vs {inputs[0]} ({sr0}). Assuming compat or process_fn handles it.")

    _safe_mkdir(os.path.dirname(os.path.abspath(output)))

    # 使用 soundfile 写出，避免一次性内存占用
    # 我们先读取第一个 chunk 来确定通道数（process_fn 可能会改变通道数）
    # 或者我们要求 process_fn 必须稳定。
    # 为了稳妥，先 peek 一下
    
    frames_per_chunk = int(chunk_dur_s * target_sr)
    total_frames = frames0 # 以第一个文件为准

    # Prime the output file writer
    sf_file = None
    
    print(f"Streaming processing to {output} (Total frames: {total_frames})...")

    cursor = 0
    while cursor < total_frames:
        current_frames = min(frames_per_chunk, total_frames - cursor)
        
        chunks_np = []
        for inp in inputs:
            # Torchaudio load returns [Channels, Frames]
            # soundfile needs [Frames, Channels]
            # 我们统一转成 [Frames, Channels] 给 process_fn
            try:
                # 容错：有些文件可能略短
                info = ta.info(inp)
                if cursor >= info.num_frames:
                    # Pad silence
                    wav = torch.zeros((info.num_channels, current_frames))
                else:
                    read_frames = min(current_frames, info.num_frames - cursor)
                    wav, _ = ta.load(inp, frame_offset=cursor, num_frames=read_frames)
                    if read_frames < current_frames:
                        # Pad end
                        pad = torch.zeros((wav.shape[0], current_frames - read_frames))
                        wav = torch.cat([wav, pad], dim=1)
                
                # 转 numpy [N, C]
                c_np = wav.transpose(0, 1).numpy().astype(np.float32)
                chunks_np.append(c_np)
            except Exception as e:
                print(f"Error reading chunk from {inp} at {cursor}: {e}")
                raise

        # Process
        if process_fn:
            out_chunk = process_fn(chunks_np, target_sr)
        else:
            out_chunk = chunks_np[0]

        # Init writer if needed
        if sf_file is None:
            channels = out_chunk.shape[1]
            sf_file = sf.SoundFile(output, mode='w', samplerate=target_sr, channels=channels)
        
        sf_file.write(out_chunk)
        
        cursor += current_frames
        if cursor % (frames_per_chunk * 10) == 0:
             print(f"  Processed {cursor / target_sr:.1f}s / {total_frames / target_sr:.1f}s")

    if sf_file:
        sf_file.close()
    
    _wait_file_stable(output)


def run_deepfilter_to_file(inp: str, out_file: str, outdir: str, timeout_s: float = 120.0):
    """
    Run deepFilter on input file. If file is too long (>30s), split into chunks to avoid VRAM OOM.
    """
    _safe_mkdir(outdir)
    _rm_tree_contents(outdir)
    _safe_mkdir(os.path.dirname(os.path.abspath(out_file)))

    sr, total_frames = _get_audio_info(inp)
    duration_s = total_frames / sr

    CHUNK_DURATION_S = 300.0 # 5 min chunks for VRAM safety
    
    # If file is short enough, process directly
    if duration_s <= CHUNK_DURATION_S:
        _run_deepfilter_process(inp, out_file, outdir, timeout_s)
        return

    # Process in chunks
    print(f"File duration {duration_s:.2f}s > {CHUNK_DURATION_S}s. Processing in chunks to save VRAM...")
    
    chunk_out_dir = os.path.join(outdir, "chunks_out")
    chunk_in_dir = os.path.join(outdir, "chunks_in")
    _safe_mkdir(chunk_out_dir)
    _safe_mkdir(chunk_in_dir)

    frames_per_chunk = int(CHUNK_DURATION_S * sr)
    
    # 记录处理后的片段路径
    processed_chunk_paths = []
    
    # 1. Split and Process Loop
    for i, start_frame in enumerate(range(0, total_frames, frames_per_chunk)):
        num_frames = min(frames_per_chunk, total_frames - start_frame)
        
        # Extract chunk
        chunk_wav, _ = ta.load(inp, frame_offset=start_frame, num_frames=num_frames)
        chunk_in_path = os.path.join(chunk_in_dir, f"chunk_{i:04d}.wav")
        # Ensure deepFilter output path knowledge
        # deepFilter outputs to a dir, filename same as input
        
        # Save input chunk
        ta.save(chunk_in_path, chunk_wav, sr)
        
        # Process chuck
        chunk_specific_out_dir = os.path.join(chunk_out_dir, f"sub_{i:04d}")
        # Predicted output file path inside that dir
        expected_out_chunk = os.path.join(chunk_specific_out_dir, f"chunk_{i:04d}.wav")
        
        # Run deepfilter
        _run_deepfilter_process(chunk_in_path, expected_out_chunk, chunk_specific_out_dir, timeout_s)
        
        processed_chunk_paths.append(expected_out_chunk)
        print(f"  Processed chunk {i+1} ({(start_frame/sr):.1f}s - {((start_frame+num_frames)/sr):.1f}s)")

    # 2. Concatenate using stream_process (Disk-based merge)
    print("Concatenating chunks (streaming)...")
    # 为了合并，我们把所有小文件当做一个“序列”拼起来
    # 但 stream_process 设计是 parallel inputs。
    # 这里是 sequential inputs。
    # 既然已经都在磁盘上了，我们可以用 sf.SoundFile 依次读写。
    
    with sf.SoundFile(out_file, mode='w', samplerate=sr, channels=processed_chunk_paths[0] and sf.info(processed_chunk_paths[0]).channels or 1) as f_out:
        for p in processed_chunk_paths:
            d, _ = sf.read(p, dtype='float32') # numpy read
            f_out.write(d)
            
    _wait_file_stable(out_file, timeout_s=timeout_s)


def _run_deepfilter_process(inp, out_file, outdir, timeout_s):
    """Helper to run a single deepFilter process"""
    _safe_mkdir(outdir)
    # _rm_tree_contents(outdir) # Dont delete if we are in a subfolder loop
    
    t_start = time.time()
    
    env = os.environ.copy()
    if "PYTORCH_CUDA_ALLOC_CONF" in env:
        del env["PYTORCH_CUDA_ALLOC_CONF"]

    tried = []
    for cmd in (
        ["deepFilter", inp, "-o", outdir],
        ["deepFilter", inp, "--output-dir", outdir],
    ):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
            created = _wait_any_wav_created(outdir, t_start, timeout_s=timeout_s)

            # copy to target
            _safe_mkdir(os.path.dirname(os.path.abspath(out_file)))
            if os.path.abspath(created) != os.path.abspath(out_file):
                shutil.copy2(created, out_file)
            _wait_file_stable(out_file, timeout_s=timeout_s)
            return
        except subprocess.CalledProcessError as e:
            tried.append((cmd, f"CalledProcessError: {e}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"))
        except Exception as e:
            tried.append((cmd, repr(e)))

    msg = "\n\n".join([f"{c}\n{err}" for c, err in tried])
    raise RuntimeError("deepFilter invocation failed. Tried:\n\n" + msg)


def peak_normalize_block(x: np.ndarray, target: float = 0.99) -> np.ndarray:
    # 块级归一化有点危险，因为不同块增益不同会导致音量跳变。
    # 真正的峰值归一化需要扫描全局。
    # 对于流式处理，通常我们做 limiter 或者保守归一化。
    # 现在的代码逻辑里，减法后的 residual 可能会爆音，所以原代码有 normalize。
    # 如果不做全局扫描，这里只能做 clip 或者 block-wise normalize。
    # 为了避免音量忽大忽小，我们只做 Soft Clip 或者 如果 > 1.0 就整体压。
    # 但如果为了安全，最好是可以接受偶尔的 clip，或者使用一个保守的 scaling 因子。
    # 旧代码：if peak > 1.0: x = x * (target / peak)
    # 这确实会导致音量呼吸效应如果分块做。
    # 妥协方案：简单 Clip，或者不做 Normalize (因为 float32 不会 clip，最后转 int16 才会)，
    # 但 saved user wav usually want no clipping.
    # 鉴于这是中间步骤，暂且不做 aggressive normalize，或者只做 hard clamp -1~1。
    # 原有的 peak_normalize 用途是防止相减后溢出。
    
    # 改进：Clamp
    return np.clip(x, -1.0, 1.0)


def suppress_speech_resonance_block(
    # inputs[0] should be input (ns_final)
    chunks: List[np.ndarray],
    sr: int,
    fmin=150,
    fmax=4000,
    reduction_db=12.0
) -> np.ndarray:
    
    audio = chunks[0]
    out = np.zeros_like(audio)

    # Short chunk processing for STFT is fine.
    
    for ch in range(audio.shape[1]):
        y = audio[:, ch]
        
        # STFT overlap-add artifacts at edges might exist but are usually minimal for bg noise tasks.
        
        S = librosa.stft(y, n_fft=2048, hop_length=512)
        mag, phase = np.abs(S), np.angle(S)

        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        mask_band = (freqs >= fmin) & (freqs <= fmax)

        # Global percentile is hard in streaming. 
        # We use local block percentile.
        time_var = np.std(mag, axis=1, keepdims=True)
        steady = time_var < np.percentile(time_var, 40)

        atten = np.ones_like(mag)
        atten[mask_band & steady.squeeze(), :] *= 10 ** (-reduction_db / 20)

        S_new = mag * atten * np.exp(1j * phase)
        y_out = librosa.istft(S_new, hop_length=512, length=len(y))

        out[:, ch] = y_out

    return out



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input vocal wav")
    ap.add_argument("--outdir", default="dfn_out", help="output directory")
    ap.add_argument("--timeout", type=float, default=120.0, help="timeout seconds for each deepFilter run")
    ap.add_argument("--keep_tmp", action="store_true", help="keep temp files")
    ap.add_argument("--export_stage1", action="store_true", help="also export non_speech_stage1.wav to outdir root")
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    tmpdir = os.path.join(outdir, "_tmp")
    df1_dir = os.path.join(tmpdir, "df1_out")
    df2_dir = os.path.join(tmpdir, "df2_out")
    _safe_mkdir(outdir)
    _safe_mkdir(tmpdir)

    stage1_speech = os.path.join(tmpdir, "speech_stage1.wav")
    stage1_ns = os.path.join(tmpdir, "non_speech_stage1.wav")
    stage2_ns_speech = os.path.join(tmpdir, "ns_speech_stage2.wav")

    out_speech = os.path.join(outdir, "speech.wav")
    out_ns = os.path.join(outdir, "non_speech.wav")
    out_ns_stage1 = os.path.join(outdir, "non_speech_stage1.wav")

    # ---------- Stage 1 ----------
    # 1. DeepFilter (Chunked internally if needed)
    print(">>> Stage 1: Running DeepFilter...")
    run_deepfilter_to_file(args.inp, stage1_speech, outdir=df1_dir, timeout_s=args.timeout)

    # 2. Subtract: Input - Stage1_Speech = Stage1_NS
    print(">>> Stage 1: Creating Non-Speech (Subtraction)...")
    
    def subtract_op(chunks, sr):
        # chunks[0] = orig, chunks[1] = speech
        # Align lengths if minor mismatch? stream_process ensures alignment by cursor reading.
        # But data might still slight mismatch in amplitude/phase alignment?
        # Assuming DeepFilter output is perfectly sample-aligned (usually is).
        diff = chunks[0] - chunks[1]
        return peak_normalize_block(diff)

    stream_process(
        inputs=[args.inp, stage1_speech],
        output=stage1_ns,
        process_fn=subtract_op,
        chunk_dur_s=30.0
    )

    if args.export_stage1:
        shutil.copy2(stage1_ns, out_ns_stage1)

    # ---------- Stage 2 ----------
    # 3. DeepFilter on Stage1_NS
    print(">>> Stage 2: Running DeepFilter on NS...")
    run_deepfilter_to_file(stage1_ns, stage2_ns_speech, outdir=df2_dir, timeout_s=args.timeout)

    # 4. Final: NS - Stage2_Speech -> Suppress -> Out_NS
    #           Input - (NS_final) -> Out_Speech
    print(">>> Stage 2: Final Processing...")

    # We need to produce TWO files: out_ns and out_speech.
    # stream_process writes one file.
    # We can run two passes, or write a custom loop here.
    # Custom loop is better to read inputs once.
    
    # Inputs: Original, Stage1_NS, Stage2_NS_Speech
    # Logic:
    #   NS_final = Stage1_NS - Stage2_NS_Speech
    #   NS_clean = suppress(NS_final)
    #   Speech_final = Original - NS_final (refined)
    
    sr, total_frames = _get_audio_info(args.inp)
    # Detect input channels from first chunk or info
    info_inp = ta.info(args.inp)
    channels = info_inp.num_channels
    
    frames_per_chunk = int(30.0 * sr)
    cursor = 0
    
    print(f"Streaming final output to {out_ns} and {out_speech} (channels={channels})...")
    
    with sf.SoundFile(out_ns, 'w', sr, channels) as f_ns, \
         sf.SoundFile(out_speech, 'w', sr, channels) as f_sp:
         
        while cursor < total_frames:
            current_frames = min(frames_per_chunk, total_frames - cursor)
            
            # Read 3 inputs
            c_orig = _read_chunk(args.inp, cursor, current_frames)
            c_ns1 = _read_chunk(stage1_ns, cursor, current_frames)
            c_ns_sp2 = _read_chunk(stage2_ns_speech, cursor, current_frames)
            
            # 1. Calc NS Final
            ns_final_chunk = c_ns1 - c_ns_sp2
            ns_final_chunk = peak_normalize_block(ns_final_chunk)
            
            # 2. Suppress Resonance
            ns_clean_chunk = suppress_speech_resonance_block([ns_final_chunk], sr)
            ns_clean_chunk = peak_normalize_block(ns_clean_chunk)
            
            # 3. Calc Speech Final
            # speech_final = orig - ns_final
            sp_final_chunk = c_orig - ns_final_chunk
            sp_final_chunk = peak_normalize_block(sp_final_chunk)
            
            # Write
            f_ns.write(ns_clean_chunk)
            f_sp.write(sp_final_chunk)
            
            cursor += current_frames
            if cursor % (frames_per_chunk * 10) == 0:
                print(f"  Processed {cursor/sr:.1f}s / {total_frames/sr:.1f}s")
                
    _wait_file_stable(out_ns)
    _wait_file_stable(out_speech)

    # Clean
    if not args.keep_tmp:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("Done:")
    print("  speech     ->", out_speech)
    print("  non_speech ->", out_ns)
    if args.export_stage1:
        print("  stage1_ns  ->", out_ns_stage1)


def _read_chunk(path, start, frames):
    # Validations omitted for brevity, assuming standard wav
    info = ta.info(path)
    if start >= info.num_frames:
        return np.zeros((frames, info.num_channels), dtype=np.float32)
    
    read_n = min(frames, info.num_frames - start)
    w, _ = ta.load(path, frame_offset=start, num_frames=read_n)
    
    # Pad if needed
    if read_n < frames:
        pad = torch.zeros((w.shape[0], frames - read_n))
        w = torch.cat([w, pad], dim=1)
        
    return w.transpose(0, 1).numpy().astype(np.float32)


if __name__ == "__main__":
    main()

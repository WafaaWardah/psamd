# wrapper for preprocessing input signals for P.SAMD

import os
import numpy as np
import torch
import torchaudio
import webrtcvad
import matplotlib.pyplot as plt


def float_to_pcm16(waveform):
    waveform = waveform.clamp(-1.0, 1.0)
    return (waveform.numpy() * 32767).astype(np.int16)


def webrtc_vad_mask(waveform, sample_rate, frame_ms=30, mode=2):
    """
    Returns a boolean speech mask per frame.
    
    mode:
        0 = least aggressive
        3 = most aggressive
    """

    vad = webrtcvad.Vad(mode)

    pcm16 = float_to_pcm16(waveform)
    bytes_audio = pcm16.tobytes()

    frame_len = int(sample_rate * frame_ms / 1000)
    bytes_per_frame = frame_len * 2  # int16 = 2 bytes

    speech_mask = []

    for i in range(0, len(bytes_audio), bytes_per_frame):
        frame = bytes_audio[i:i + bytes_per_frame]

        if len(frame) < bytes_per_frame:
            break

        is_speech = vad.is_speech(frame, sample_rate)
        speech_mask.append(is_speech)

    speech_mask = torch.tensor(speech_mask, dtype=torch.bool)

    return speech_mask, frame_len


def trim_leading_trailing_silence(
    waveform,
    speech_mask,
    frame_len,
    hop_len,
    sample_rate,
    margin_ms=150,
):
    """
    Trim leading and trailing non-speech while keeping a small margin.
    """

    speech_indices = torch.where(speech_mask)[0]

    # If no speech detected, return unchanged
    if speech_indices.numel() == 0:
        return waveform

    margin_samples = int(sample_rate * margin_ms / 1000)

    first_frame = int(speech_indices[0])
    last_frame = int(speech_indices[-1])

    start_sample = max(0, first_frame * hop_len - margin_samples)
    end_sample = min(
        waveform.numel(),
        last_frame * hop_len + frame_len + margin_samples
    )

    return waveform[start_sample:end_sample]


def reduce_long_internal_pauses(
    waveform,
    speech_mask,
    frame_len,
    sample_rate,
    max_pause_ms=300,
):
    """
    Reduce long internal non-speech pauses while preserving active speech.

    Assumes WebRTC VAD mask with non-overlapping frames.
    waveform: torch.Tensor [T]
    speech_mask: torch.BoolTensor [num_frames]
    frame_len: samples per VAD frame
    """

    max_pause_frames = max(1, int(max_pause_ms / (1000 * frame_len / sample_rate)))

    pieces = []
    n_frames = len(speech_mask)

    i = 0
    while i < n_frames:
        frame_start = i * frame_len
        frame_end = min((i + 1) * frame_len, waveform.numel())

        if speech_mask[i]:
            pieces.append(waveform[frame_start:frame_end])
            i += 1
        else:
            # start of silence region
            silence_start = i

            while i < n_frames and not speech_mask[i]:
                i += 1

            silence_end = i  # exclusive

            has_speech_before = silence_start > 0 and speech_mask[silence_start - 1]
            has_speech_after = silence_end < n_frames and speech_mask[silence_end]

            # Only reduce internal pauses, not leading/trailing silence
            if has_speech_before and has_speech_after:
                keep_frames = min(silence_end - silence_start, max_pause_frames)

                keep_start = silence_start * frame_len
                keep_end = min((silence_start + keep_frames) * frame_len, waveform.numel())

                pieces.append(waveform[keep_start:keep_end])
            else:
                # keep leading/trailing silence unchanged here
                pieces.append(waveform[frame_start:min(silence_end * frame_len, waveform.numel())])

    if not pieces:
        return waveform

    return torch.cat(pieces)


def plot_waveform_and_spectrogram(waveform, sample_rate, title=""):
    """
    waveform: torch.Tensor [T]
    """

    waveform_np = waveform.cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # ---- Waveform ----
    time = np.arange(len(waveform_np)) / sample_rate
    axes[0].plot(time, waveform_np)
    axes[0].set_title(f"{title} - Waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    # ---- Spectrogram (log-mel via torchaudio) ----
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=480,   # 10 ms at 48 kHz
        n_mels=128
    )(waveform)

    log_mel = torch.log(mel_spec + 1e-10)

    axes[1].imshow(
        log_mel.cpu().numpy(),
        aspect='auto',
        origin='lower'
    )
    axes[1].set_title(f"{title} - Log-Mel Spectrogram")
    axes[1].set_xlabel("Time Frames")
    axes[1].set_ylabel("Mel Bins")

    plt.tight_layout()
    plt.show()



def preprocess(waveform, sample_rate, target_sr=48000):
    """
    Preprocess input signal before P.SAMD inference.

    Constraints aimed for:
    - target sample rate: 48 kHz
    - final duration: 6-8 s
    - leading/trailing pause: about 100-200 ms
    - no long internal pauses
    - speech activity: preferably >70%

    Args:
        waveform: torch.Tensor [T] or [C, T]
        sample_rate: int
        target_sr: int

    Returns:
        waveform: torch.Tensor [T]
        sample_rate: int
    """

    # 1. Ensure waveform is float tensor
    waveform = waveform.float()

    # 2. Convert stereo/multi-channel to mono if needed
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
        #print("Converted multi-channel input to mono.")

    # ---- BEFORE ----
    #plot_waveform_and_spectrogram(waveform, sample_rate, title="Before Wrapper")

    # 3. Resample to target sampling rate
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sr
        )
        waveform = resampler(waveform)
        sample_rate = target_sr
        #print(f"Resampled input from {sample_rate} Hz to {target_sr} Hz.")

    # 4. Estimate speech activity
    speech_mask, frame_len = webrtc_vad_mask(
        waveform,
        sample_rate,
        frame_ms=30,
        mode=2
    )

    speech_ratio = speech_mask.float().mean().item()
    #print(f"\nSpeech activity ratio (WebRTC VAD): {speech_ratio:.2f}")

    duration_s = waveform.numel() / sample_rate
    #print(f"Duration before leading/trailing trim: {duration_s:.2f} s")

    # 5. Trim leading/trailing silence while keeping ~150 ms margin
    hop_len = frame_len  # since frames are non-overlapping here
    waveform = trim_leading_trailing_silence(
        waveform,
        speech_mask,
        frame_len,
        hop_len,
        sample_rate,
        margin_ms=150,
    )

    # Recompute speech activity after trimming
    # 4. Estimate speech activity
    speech_mask, frame_len = webrtc_vad_mask(
        waveform,
        sample_rate,
        frame_ms=30,
        mode=2
    )

    speech_ratio = speech_mask.float().mean().item()
    #print(f"\nSpeech activity ratio (WebRTC VAD): {speech_ratio:.2f}")

    duration_s = waveform.numel() / sample_rate
    #print(f"Duration after leading/trailing trim: {duration_s:.2f} s")

    # 6. Check whether the processed signal meets the proposed constraints
    duration_s = waveform.numel() / sample_rate
    speech_ratio = speech_mask.float().mean().item()

    meets_duration = 6.0 <= duration_s <= 10.0
    meets_speech_ratio = speech_ratio >= 0.70

    #print(f"\nMeets duration constraint (6-10 s): {meets_duration}")
    #print(f"Meets speech activity constraint (>0.70): {meets_speech_ratio}")

    # 7. Reduce long internal pauses
    speech_mask, frame_len = webrtc_vad_mask(
        waveform,
        sample_rate,
        frame_ms=30,
        mode=2
    )

    waveform = reduce_long_internal_pauses(
        waveform,
        speech_mask,
        frame_len,
        sample_rate,
        max_pause_ms=300,
    )

    # Recompute speech activity and duration after internal pause reduction
    speech_mask, frame_len = webrtc_vad_mask(
        waveform,
        sample_rate,
        frame_ms=30,
        mode=2
    )


    speech_ratio = speech_mask.float().mean().item()
    duration_s = waveform.numel() / sample_rate

    if duration_s < 6.0 or duration_s > 10.0 or speech_ratio < 0.70:
        print(f"flag")
        

    #print(f"Speech activity ratio after internal pause reduction: {speech_ratio:.2f}")
    #print(f"Duration after internal pause reduction: {duration_s:.2f} s")

    # ---- AFTER ----
    #plot_waveform_and_spectrogram(waveform, sample_rate, title="After Wrapper")

    return waveform, sample_rate
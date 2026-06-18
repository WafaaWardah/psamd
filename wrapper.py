# wrapper.py version 3 - Wafaa Wardah, TU-Berlin, 2026
# Preprocessing wrapper for P.SAMD based Annex C, and intended for use with audio files 
# containing speech with internal pauses. Ideal audio file should be over 8 seconds long and 
# contain at least 5.5 seconds of active speech, but the wrapper will attempt to select the 
# best segment according to the criteria described below. 
# See ITU-T P.P566 Annex C for details, version: SG12-TD473 June 2026
#
import os
import numpy as np
import torch
import torchaudio
import webrtcvad


def float_to_pcm16(waveform: torch.Tensor) -> np.ndarray:
    waveform = waveform.detach().cpu().float().clamp(-1.0, 1.0)
    return (waveform.numpy() * 32767).astype(np.int16)


def webrtc_vad_mask(
    waveform: torch.Tensor,
    sample_rate: int,
    frame_ms: int = 30,
    mode: int = 2,
):
    
    if sample_rate not in (8000, 16000, 32000, 48000):
        raise ValueError("WebRTC VAD only supports 8, 16, 32, or 48 kHz.")

    if frame_ms not in (10, 20, 30):
        raise ValueError("WebRTC VAD frame duration must be 10, 20, or 30 ms.")

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

        speech_mask.append(vad.is_speech(frame, sample_rate))

    return torch.tensor(speech_mask, dtype=torch.bool), frame_len


def prepare_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sr: int = 48000,
):
   
    waveform = waveform.float()

    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)

    if waveform.ndim != 1:
        raise ValueError("Expected waveform shape [T] or [C, T].")

    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sr
        )
        waveform = resampler(waveform)
        sample_rate = target_sr

    return waveform, sample_rate


def find_active_speech_segment(
    info: dict,
    waveform: torch.Tensor,
    speech_mask: torch.Tensor,
    frame_len: int,
    sample_rate: int,
    pre_margin_ms: int = 150,
    min_active_speech_s: float = 5.5,
    max_active_speech_s: float = 8.0,
    max_internal_pause_s: float = 2.0,
    min_stop_pause_ms: int = 90,
):

    speech_indices = torch.where(speech_mask)[0]
    
    if speech_indices.numel() == 0:
        info["flag"] = True
        info["reason"] = "no_speech_detected"
        info["reject_end_sample"] = waveform.numel()
        return None, info

    n_frames = len(speech_mask)
    frame_duration_s = frame_len / sample_rate

    pre_margin_samples = int(sample_rate * pre_margin_ms / 1000)
    min_stop_pause_frames = max(1, int(round(min_stop_pause_ms / (1000 * frame_duration_s))))
    max_internal_pause_frames = max(1, int(round(max_internal_pause_s / frame_duration_s)))
    max_segment_samples = int(max_active_speech_s * sample_rate)

    first_speech_frame = int(speech_indices[0])
    start_sample = max(0, first_speech_frame * frame_len - pre_margin_samples)

    active_speech_s = 0.0
    seen_speech = False
    silence_run = 0
    end_frame = None

    i = first_speech_frame

    while i < n_frames:
        current_sample = min((i + 1) * frame_len, waveform.numel())
        current_segment_duration = current_sample - start_sample

        if speech_mask[i]:
            seen_speech = True
            active_speech_s += frame_duration_s
            silence_run = 0

            if active_speech_s >= max_active_speech_s:
                info["flag"] = True
                info["reason"] = "max_active_speech_reached_without_stop_pause"
                info["reject_end_sample"] = min((i + 1) * frame_len, waveform.numel())
                break

            i += 1
            continue


        # Non-speech frame
        if seen_speech:
            silence_start = i

            while i < n_frames and not speech_mask[i]:
                silence_run += 1
                i += 1

                if silence_run > max_internal_pause_frames:
                    info["flag"] = True
                    info["reason"] = "internal_pause_too_long"
                    info["reject_end_sample"] = min(i * frame_len, waveform.numel())
                    break

            if info["flag"]:
                break

            pause_len_frames = i - silence_start

            if (
                active_speech_s >= min_active_speech_s
                and pause_len_frames >= min_stop_pause_frames
            ):
                end_frame = silence_start
                break

            silence_run = 0
        else:
            i += 1

    if end_frame is None and not info["flag"]:
        info["flag"] = True
        info["reason"] = "insufficient_active_speech_or_no_pause_found"
        info["reject_end_sample"] = waveform.numel()

    if info["flag"]:
        info["active_speech_s"] = active_speech_s
        return None, info

    end_sample = min(end_frame * frame_len, waveform.numel())

    if end_sample <= start_sample:
        info["flag"] = True
        info["reason"] = "invalid_segment_boundaries"
        info["active_speech_s"] = active_speech_s
        info["reject_end_sample"] = max(1, end_sample)
        return None, info

    segment = waveform[start_sample:end_sample]

    # Diagnostics for selected segment
    segment_mask, _ = webrtc_vad_mask(segment, sample_rate, frame_ms=30, mode=2)
    speech_ratio = segment_mask.float().mean().item() if len(segment_mask) > 0 else 0.0

    info.update({
        "flag": False,
        "reason": "ok",
        "start_sample": start_sample,
        "end_sample": end_sample,
        "segment_duration_s": segment.numel() / sample_rate,
        "active_speech_s": active_speech_s,
        "speech_ratio": speech_ratio,
    })

    return segment, info


def preprocess(
    waveform: torch.Tensor,
    data_dir: str,
    new_dir: str,
    sample_rate: int,
    target_sr: int = 48000,
    frame_ms: int = 30,
    vad_mode: int = 2,
    pre_margin_ms: int = 150,
    min_active_speech_s: float = 5.5,
    max_active_speech_s: float = 8.0,
    max_internal_pause_s: float = 2.0,
    min_stop_pause_ms: int = 90,
    return_info: bool = False,
):

    waveform, sample_rate = prepare_waveform(
        waveform=waveform,
        sample_rate=sample_rate,
        target_sr=target_sr,
    )

    remaining_waveform = waveform
    seg_count = 0
    cursor_sample = 0
    all_info = []

    # text file to log segment info
    log_path = os.path.join(new_dir, "segment_info.txt")
    with open(log_path, "a") as log_file:

        while True:

            if remaining_waveform.numel() < sample_rate:
                #print("Less than 1 second remains, stopping.")
                break

            speech_mask, frame_len = webrtc_vad_mask(
                waveform=remaining_waveform,
                sample_rate=sample_rate,
                frame_ms=frame_ms,
                mode=vad_mode,
            )

            info = {
                "flag": False,
                "reason": "ok",
                "start_sample": None,
                "end_sample": None,
                "segment_duration_s": None,
                "active_speech_s": 0.0,
                "speech_ratio": None,
            }

            segment, info = find_active_speech_segment(
                info=info,
                waveform=remaining_waveform,
                speech_mask=speech_mask,
                frame_len=frame_len,
                sample_rate=sample_rate,
                pre_margin_ms=pre_margin_ms,
                min_active_speech_s=min_active_speech_s,
                max_active_speech_s=max_active_speech_s,
                max_internal_pause_s=max_internal_pause_s,
                min_stop_pause_ms=min_stop_pause_ms,
            )

            if segment is None or info["flag"]:
                #print("Rejected candidate segment:")
                #print(info)

                skip_samples = info.get("reject_end_sample")

                if skip_samples is None:
                    raise RuntimeError(
                        f"Rejected segment without reject_end_sample. "
                        f"Reason: {info.get('reason')}"
                    )

                if skip_samples <= 0:
                    raise RuntimeError(
                        f"Invalid reject_end_sample={skip_samples}. "
                        f"Reason: {info.get('reason')}"
                    )

                if remaining_waveform.numel() <= skip_samples:
                    #print("No more audio to scan, stopping.")
                    break

                cursor_sample += skip_samples
                remaining_waveform = remaining_waveform[skip_samples:]
                continue

            if info["end_sample"] is None or info["end_sample"] <= 0:
                #print("No forward progress possible, stopping.")
                #print(info)
                break

            seg_count += 1

            #print(info)
            log_file.write(f"File {os.path.basename(data_dir)} - Segment {seg_count}:\n")
            for key, value in info.items():
                log_file.write(f"  {key}: {value}, ")
            log_file.write("\n")


            segment_path = os.path.join(
                new_dir,
                f"{os.path.basename(data_dir)}_seg_{seg_count}.wav"
            )

            torchaudio.save(segment_path, segment.unsqueeze(0), sample_rate)

            info["global_start_sample"] = cursor_sample + info["start_sample"]
            info["global_end_sample"] = cursor_sample + info["end_sample"]
            info["segment_path"] = segment_path
            all_info.append(info)

            cursor_sample += info["end_sample"]
            remaining_waveform = remaining_waveform[info["end_sample"]:]

        if return_info:
            print(all_info)
            log_file.write("All segments info:\n")
            for idx, info in enumerate(all_info):
                log_file.write(f"Segment {idx}:\n")
                for key, value in info.items():
                    log_file.write(f"  {key}: {value}\n")
                log_file.write("\n")


def post_mapping(y):
    y_mapped = -1.0 + 1.2 * y

    if hasattr(y_mapped, "clamp"):
        return y_mapped.clamp(1.0, 5.0)

    return y_mapped.clip(1.0, 5.0)

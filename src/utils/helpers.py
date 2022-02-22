import librosa
import matplotlib.pylab as plt
import numpy as np
import torch
import wandb
from nemo.utils import logging

try:
    from pytorch_lightning.utilities import rank_zero_only
except ModuleNotFoundError:
    from functools import wraps

    def rank_zero_only(fn):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            logging.error(
                f"Function {fn} requires lighting to be installed, but it was not found. Please install lightning first"
            )
            exit(1)


def griffin_lim(magnitudes, n_iters=50, n_fft=1024):
    """
    Griffin-Lim algorithm to convert magnitude spectrograms to audio signals
    """
    phase = np.exp(2j * np.pi * np.random.rand(*magnitudes.shape))
    complex_spec = magnitudes * phase
    signal = librosa.istft(complex_spec)
    if not np.isfinite(signal).all():
        logging.warning("audio was not finite, skipping audio saving")
        return np.array([0])

    for _ in range(n_iters):
        _, phase = librosa.magphase(librosa.stft(signal, n_fft=n_fft))
        complex_spec = magnitudes * phase
        signal = librosa.istft(complex_spec)
    return signal


def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_spectrogram_to_numpy(spectrogram):
    spectrogram = spectrogram.astype(np.float32)
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_gate_outputs_to_numpy(gate_targets, gate_outputs):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(
        range(len(gate_targets)),
        gate_targets,
        alpha=0.5,
        color="green",
        marker="+",
        s=1,
        label="target",
    )
    ax.scatter(
        range(len(gate_outputs)),
        gate_outputs,
        alpha=0.5,
        color="red",
        marker=".",
        s=1,
        label="predicted",
    )

    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("Gate State")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


@rank_zero_only
def tacotron2_log_to_tb_func(
    swriter,
    tensors,
    step,
    tag="train",
    log_images=False,
    log_images_freq=1,
    add_audio=True,
    griffin_lim_mag_scale=1024,
    griffin_lim_power=1.2,
    sr=22050,
    n_fft=1024,
    n_mels=80,
    fmax=8000,
    sample_idx=0,
):
    _, spec_target, mel_postnet, gate, gate_target, alignments = tensors
    if log_images and step % log_images_freq == 0:
        swriter.add_image(
            f"{tag}_alignment",
            plot_alignment_to_numpy(alignments[sample_idx].data.cpu().numpy().T),
            step,
            dataformats="HWC",
        )
        swriter.add_image(
            f"{tag}_mel_target",
            plot_spectrogram_to_numpy(spec_target[sample_idx].data.cpu().numpy()),
            step,
            dataformats="HWC",
        )
        swriter.add_image(
            f"{tag}_mel_predicted",
            plot_spectrogram_to_numpy(mel_postnet[sample_idx].data.cpu().numpy()),
            step,
            dataformats="HWC",
        )
        swriter.add_image(
            f"{tag}_gate",
            plot_gate_outputs_to_numpy(
                gate_target[sample_idx].data.cpu().numpy(),
                torch.sigmoid(gate[sample_idx]).data.cpu().numpy(),
            ),
            step,
            dataformats="HWC",
        )
        if add_audio:
            filterbank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax)
            log_mel = mel_postnet[sample_idx].data.cpu().numpy().T
            mel = np.exp(log_mel)
            magnitude = np.dot(mel, filterbank) * griffin_lim_mag_scale
            audio = griffin_lim(magnitude.T ** griffin_lim_power, n_fft=n_fft)
            swriter.add_audio(f"audio/{tag}_predicted", audio / max(np.abs(audio)), step, sample_rate=sr)

            log_mel = spec_target[sample_idx].data.cpu().numpy().T
            mel = np.exp(log_mel)
            magnitude = np.dot(mel, filterbank) * griffin_lim_mag_scale
            audio = griffin_lim(magnitude.T ** griffin_lim_power, n_fft=n_fft)
            swriter.add_audio(f"audio/{tag}_target", audio / max(np.abs(audio)), step, sample_rate=sr)


@rank_zero_only
def transformer_tts_log_to_tb_func(
    swriter,
    tensors,
    step,
    tag="train",
    log_images=False,
    log_images_freq=1,
    add_audio=True,
    griffin_lim_mag_scale=1024,
    griffin_lim_power=1.2,
    sr=22050,
    n_fft=1024,
    n_mels=80,
    fmax=8000,
    sample_idx=0,
):
    _, spec_target, mel_postnet, gate, gate_target, alignments = tensors
    spec_target = spec_target.transpose(1, 2)
    mel_postnet = mel_postnet.transpose(1, 2)
    if log_images and step % log_images_freq == 0:
        for i in range(alignments.shape[1]):
            swriter.add_image(
                f"{tag}_alignment_head_{i}",
                plot_alignment_to_numpy(alignments[sample_idx, i].data.cpu().numpy().T),
                step,
                dataformats="HWC",
            )
        swriter.add_image(
            f"{tag}_mel_target",
            plot_spectrogram_to_numpy(spec_target[sample_idx].data.cpu().numpy()),
            step,
            dataformats="HWC",
        )
        swriter.add_image(
            f"{tag}_mel_predicted",
            plot_spectrogram_to_numpy(mel_postnet[sample_idx].data.cpu().numpy()),
            step,
            dataformats="HWC",
        )
        swriter.add_image(
            f"{tag}_gate",
            plot_gate_outputs_to_numpy(
                gate_target[sample_idx].data.cpu().numpy(),
                torch.sigmoid(gate[sample_idx]).data.cpu().numpy(),
            ),
            step,
            dataformats="HWC",
        )
        if add_audio:
            filterbank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax)
            log_mel = mel_postnet[sample_idx].data.cpu().numpy().T
            mel = np.exp(log_mel)
            magnitude = np.dot(mel, filterbank) * griffin_lim_mag_scale
            audio = griffin_lim(magnitude.T ** griffin_lim_power, n_fft=n_fft)
            swriter.add_audio(f"audio/{tag}_predicted", audio / max(np.abs(audio)), step, sample_rate=sr)

            log_mel = spec_target[sample_idx].data.cpu().numpy().T
            mel = np.exp(log_mel)
            magnitude = np.dot(mel, filterbank) * griffin_lim_mag_scale
            audio = griffin_lim(magnitude.T ** griffin_lim_power, n_fft=n_fft)
            swriter.add_audio(f"audio/{tag}_target", audio / max(np.abs(audio)), step, sample_rate=sr)

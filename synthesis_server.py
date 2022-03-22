import base64
from pathlib import Path

import flask
import torch
import torchaudio
from flask_cors import CORS
from PIL import Image

from src.utils.eval_spectrogram_generator import load_nemo_model
from src.utils.helpers import plot_spectrogram_to_numpy

app = flask.Flask(__name__)
CORS(app)


vocoder = load_nemo_model(
    Path("/home/maxim/synthesis/multispeaker-synthesis/models/hifigan--val_loss=0.17-epoch=749-last.ckpt")
)
acoustic_model = load_nemo_model(
    Path(
        "/home/maxim/synthesis/multispeaker-synthesis/models/tacotron2_hifi_singlespeaker--val_loss=8.58-epoch=439-last.ckpt"
    )
)


def get_file_name(root_path: Path, ext: str = ".jpg"):
    num_files = len(list(root_path.iterdir()))

    return root_path / f"{num_files}{ext}"


@app.route("/", methods=["GET"])
def json_example():
    input_text = flask.request.args.get("text")
    print(f"TEXT: {input_text}")

    with torch.no_grad():
        tokens = acoustic_model.parse(input_text)
        spec = acoustic_model.generate_spectrogram(tokens=tokens)
        out_wav = vocoder(spec=spec)
    print(f"TOKENS: {tokens}")
    print(f"WAV SAMPLES: {out_wav.shape}")

    img_np = plot_spectrogram_to_numpy(spec.squeeze().data.cpu().numpy())
    im = Image.fromarray(img_np)
    img_filename = get_file_name(Path("cached_images"), ".jpg")
    im.save(img_filename)

    wav_filename = get_file_name(Path("cached_wavs"), ".wav")
    torchaudio.save(wav_filename, out_wav.squeeze(0).data.cpu(), 22050)

    with open(img_filename, "rb") as file:
        image_binary = file.read()
    with open(wav_filename, "rb") as file:
        wav_binary = file.read()

    image = base64.b64encode(image_binary).decode("utf-8")
    audio = base64.b64encode(wav_binary).decode("utf-8")

    return flask.jsonify({"status": True, "image": image, "audio": audio})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=54321)

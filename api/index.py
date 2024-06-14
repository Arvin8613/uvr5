import soundfile as sf
import torch 
import os 
import librosa
import numpy as np
import onnxruntime as ort
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from flask import Flask, request, jsonify
from flask_cors import CORS

class ConvTDFNet:
    def __init__(self, target_name, L, dim_f, dim_t, n_fft, hop=1024):
        super(ConvTDFNet, self).__init__()
        self.dim_c = 4
        self.dim_f = dim_f
        self.dim_t = 2**dim_t
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.target_name = target_name
        
        out_c = self.dim_c * 4 if target_name == "*" else self.dim_c
        
        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t])
        self.n = L // 2

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, self.dim_c, self.n_bins, self.dim_t]
        )
        return x[:, :, : self.dim_f]

    # Inversed Short-time Fourier transform (STFT).
    def istft(self, x, freq_pad=None):
        freq_pad = (
            self.freq_pad.repeat([x.shape[0], 1, 1, 1])
            if freq_pad is None
            else freq_pad
        )
        x = torch.cat([x, freq_pad], -2)
        c = 4 * 2 if self.target_name == "*" else 2
        x = x.reshape([-1, c, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(
            x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True
        )
        return x.reshape([-1, c, self.chunk_size])

class Predictor:
    def __init__(self, model_path, dim_f, dim_t, n_fft, denoise=True, margin=44100, chunks=15):
        self.model_ = ConvTDFNet(
            target_name="vocals",
            L=11,
            dim_f=dim_f,
            dim_t=dim_t,
            n_fft=n_fft
        )
        self.denoise = denoise
        self.model_path = model_path
        self.margin = margin
        self.chunks = chunks
        if torch.cuda.is_available():
            self.model = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider'])
        else:
            self.model = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])

    def demix(self, mix):
        samples = mix.shape[-1]
        margin = self.margin
        chunk_size = self.chunks * 44100
        
        assert not margin == 0, "margin cannot be zero!"
        
        if margin > chunk_size:
            margin = chunk_size

        segmented_mix = {}

        if self.chunks == 0 or samples < chunk_size:
            chunk_size = samples

        counter = -1
        for skip in range(0, samples, chunk_size):
            counter += 1
            s_margin = 0 if counter == 0 else margin
            end = min(skip + chunk_size + margin, samples)
            start = skip - s_margin
            segmented_mix[skip] = mix[:, start:end].copy()
            if end == samples:
                break

        sources = self.demix_base(segmented_mix, margin_size=margin)
        return sources

    def demix_base(self, mixes, margin_size):
        chunked_sources = []
        progress_bar = tqdm(total=len(mixes))
        progress_bar.set_description("Processing")    

        for mix in mixes:
            cmix = mixes[mix]
            sources = []
            n_sample = cmix.shape[1]
            model = self.model_
            trim = model.n_fft // 2
            gen_size = model.chunk_size - 2 * trim
            pad = gen_size - n_sample % gen_size
            mix_p = np.concatenate(
                (np.zeros((2, trim)), cmix, np.zeros((2, pad)), np.zeros((2, trim))), 1
            )
            mix_waves = []
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i : i + model.chunk_size])
                mix_waves.append(waves)
                i += gen_size
            
            mix_waves = torch.tensor(np.array(mix_waves), dtype=torch.float32)
            
            with torch.no_grad():
                _ort = self.model
                spek = model.stft(mix_waves)
                if self.denoise:
                    spec_pred = (
                        -_ort.run(None, {"input": -spek.cpu().numpy()})[0] * 0.5
                        + _ort.run(None, {"input": spek.cpu().numpy()})[0] * 0.5
                    )
                    tar_waves = model.istft(torch.tensor(spec_pred))
                else:
                    tar_waves = model.istft(
                        torch.tensor(_ort.run(None, {"input": spek.cpu().numpy() })[0])
                    )
                tar_signal = (
                    tar_waves[:, :, trim:-trim]
                    .transpose(0, 1)
                    .reshape(2, -1)
                    .numpy()[:, :-pad]
                )

                start = 0 if mix == 0 else margin_size
                end = None if mix == list(mixes.keys())[::-1][0] else -margin_size
 
                if margin_size == 0:
                    end = None
                
                sources.append(tar_signal[:, start:end])

                progress_bar.update(1)

            chunked_sources.append(sources)
        _sources = np.concatenate(chunked_sources, axis=-1)

        progress_bar.close()
        return _sources

    def predict(self, file_path):
        mix, rate = librosa.load(file_path, mono=False, sr=44100)

        if mix.ndim == 1:
            mix = np.asfortranarray([mix, mix])

        mix = mix.T
        sources = self.demix(mix.T)
        opt = sources[0].T

        return (mix - opt, opt, rate)

print(os.getcwd())
predictor = Predictor(
        model_path=os.path.join(os.getcwd(), 'UVR-MDX-NET-Inst_HQ_3.onnx'),
        dim_f=3072,
        dim_t=8,
        n_fft=6144,
        denoise=True,
        margin=44100,
        chunks=15
    )
def separate_audio(file_path, output_dir):
    vocals, no_vocals, sampling_rate = predictor.predict(file_path)
    filename = os.path.splitext(os.path.split(file_path)[-1])[0]
    sf.write(os.path.join(output_dir, filename + "_no_vocals.wav"), no_vocals, sampling_rate)
    sf.write(os.path.join(output_dir, filename + "_vocals.wav"), vocals, sampling_rate)

    return os.path.join(output_dir, filename + "_no_vocals.wav"), os.path.join(output_dir, filename + "_vocals.wav")


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # 启用CORS，允许所有来源

@app.route('/uvr5', methods=['POST', 'OPTIONS'])
def post_endpoint():
    if request.method == 'OPTIONS':
        # 处理预检请求
        response = app.make_default_options_response()
        headers = response.headers

        headers['Access-Control-Allow-Origin'] = '*'
        headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'

        return response

    if request.content_type != 'application/json':
        return jsonify({'error': 'Content-Type must be application/json'}), 415
    
    try:
        data = request.json
        if data is None:
            return jsonify({'error': 'Invalid JSON data'}), 400

        # Example of accessing a specific attribute
        if 'file_path' in data:
            file_path = data['file_path']
        else:
            return jsonify({'error': 'file_path is required'}), 400
     
        no_vocal, vocal = separate_audio(file_path=data['file_path'], output_dir=data['output_dir'])
        return jsonify({
            'vocal': vocal,
            'no_vocal': no_vocal
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
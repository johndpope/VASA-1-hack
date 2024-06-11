
import soundfile as sf
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2Processor
class Wav2VecFeatureExtractor:
    def __init__(self, model_name='facebook/wav2vec2-base-960h', device='cpu'):
        self.model_name = model_name
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(device)

    def extract_features_from_wav(self, audio_path, m=2, n=2):
            """
            Extract audio features from a WAV file using Wav2Vec 2.0.

            Args:
                audio_path (str): Path to the WAV audio file.
                m (int): The number of frames before the current frame to include.
                n (int): The number of frames after the current frame to include.

            Returns:
                torch.Tensor: Features extracted from the audio for each frame.
            """
            # Load the audio file
            waveform, sample_rate = sf.read(audio_path)

            # Check if we need to resample
            if sample_rate != self.processor.feature_extractor.sampling_rate:
                waveform = librosa.resample(np.float32(waveform), orig_sr=sample_rate, target_sr=self.processor.feature_extractor.sampling_rate)
                sample_rate = self.processor.feature_extractor.sampling_rate

            # Ensure waveform is a 1D array for a single-channel audio
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)  # Taking mean across channels for simplicity

            # Process the audio to extract features
            input_values = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values
            input_values = input_values.to(self.device)

            # Pass the input_values to the model
            with torch.no_grad():
                hidden_states = self.model(input_values).last_hidden_state

            num_frames = hidden_states.shape[1]
            feature_dim = hidden_states.shape[2]

            # Concatenate nearby frame features
            all_features = []
            for f in range(num_frames):
                start_frame = max(f - m, 0)
                end_frame = min(f + n + 1, num_frames)
                frame_features = hidden_states[0, start_frame:end_frame, :].flatten()

                # Add padding if necessary
                if f - m < 0:
                    front_padding = torch.zeros((m - f) * feature_dim, device=self.device)
                    frame_features = torch.cat((front_padding, frame_features), dim=0)
                if f + n + 1 > num_frames:
                    end_padding = torch.zeros(((f + n + 1 - num_frames) * feature_dim), device=self.device)
                    frame_features = torch.cat((frame_features, end_padding), dim=0)

                all_features.append(frame_features)

            all_features = torch.stack(all_features, dim=0)
            return all_features
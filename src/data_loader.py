import os
import warnings

import mne
import numpy as np
import torch
from mne.io import read_raw_gdf
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')


class BCIDataset:
    def __init__(self, data_path, subject_id):
        self.data_path = data_path
        self.subject_id = subject_id
        self.sampling_freq = 250
        self.n_channels = 22

        # Standard electrode positions
        self.channel_names = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'P1', 'Pz', 'P2', 'POz'
        ]

    def load_data(self):
        filename = f"A{self.subject_id:02d}T.gdf"
        filepath = os.path.join(self.data_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        raw = read_raw_gdf(filepath, preload=True,
                           verbose=False, stim_channel='auto')

        # Keep only the first 22 EEG channels
        if len(raw.ch_names) > self.n_channels:
            raw.pick(raw.ch_names[:self.n_channels])
        else:
            raw.pick_types(eeg=True)

        # Rename to standard names
        mapping = dict(zip(raw.ch_names, self.channel_names))
        raw.rename_channels(mapping)
        raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})

        return raw


class EEGProcessor:
    def __init__(self, low_freq=8, high_freq=30, use_filterbank=False):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.use_filterbank = use_filterbank
        self.fs = 250

        # Common frequency bands for MI-BCI
        self.filter_bands = [
            (8, 12),
            (12, 16),
            (16, 20),
            (20, 24),
            (24, 28),
            (28, 35)
        ] if use_filterbank else [(low_freq, high_freq)]

        self.event_mapping = {
            769: 'left_hand',
            770: 'right_hand',
            771: 'feet',
            772: 'tongue'
        }

        self.label_mapping = {
            'left_hand': 0,
            'right_hand': 1,
            'feet': 2,
            'tongue': 3
        }

    def preprocess(self, raw):
        # Set montage for spatial filtering
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage, on_missing='ignore')
        except:
            pass

        # Basic preprocessing pipeline
        # Remove powerline noise
        raw.notch_filter(freqs=50, picks='eeg', verbose=False)
        raw.filter(l_freq=0.5, h_freq=None, picks='eeg',
                   verbose=False)  # Highpass
        raw.set_eeg_reference('average', projection=True, verbose=False)  # CAR
        raw.apply_proj(verbose=False)

        return raw

    def extract_epochs(self, raw):
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # Map events to motor imagery tasks
        mi_events = {}
        for code, name in self.event_mapping.items():
            if str(code) in event_id:
                mi_events[name] = event_id[str(code)]
            elif code in event_id.values():
                for k, v in event_id.items():
                    if v == code:
                        mi_events[name] = v
                        break

        if not mi_events:
            raise ValueError("No motor imagery events found")

        # Filter events
        valid_events = []
        for event in events:
            if event[2] in mi_events.values():
                valid_events.append(event)
        valid_events = np.array(valid_events)

        # Create epochs: 0.5s to 4.5s after cue (4 seconds of MI)
        epochs = mne.Epochs(raw, valid_events, mi_events, tmin=0.5, tmax=4.5,
                            baseline=None, preload=True, proj=True, verbose=False)

        return epochs

    def extract_features(self, epochs):
        data = epochs.get_data()

        # Apply frequency filtering
        filtered_data = []
        for trial in data:
            trial_bands = []
            for low, high in self.filter_bands:
                filtered = self._bandpass_filter(trial, low, high)
                trial_bands.append(filtered)
            filtered_data.append(np.array(trial_bands))

        X = np.array(filtered_data)
        y = self._get_labels(epochs)

        return X, y

    def _bandpass_filter(self, data, low_freq, high_freq):
        nyquist = self.fs / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = butter(4, [low, high], btype='band')

        filtered = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered[ch] = filtfilt(b, a, data[ch])

        return filtered

    def _get_labels(self, epochs):
        labels = []
        for event in epochs.events:
            event_code = event[2]
            found = False

            for name, code in epochs.event_id.items():
                if code == event_code and name in self.label_mapping:
                    labels.append(self.label_mapping[name])
                    found = True
                    break

            if not found:
                labels.append(0)  # Default to class 0

        return np.array(labels)

    def normalize_data(self, X):
        X_norm = np.zeros_like(X)

        for trial in range(X.shape[0]):
            for band in range(X.shape[1]):
                for ch in range(X.shape[2]):
                    data = X[trial, band, ch, :]
                    mean_val = np.mean(data)
                    std_val = np.std(data)

                    if std_val > 0:
                        X_norm[trial, band, ch, :] = (
                            data - mean_val) / std_val
                    else:
                        X_norm[trial, band, ch, :] = data

        return X_norm


class MotorImageryDataset(Dataset):
    def __init__(self, X, y, normalize=True, augment=False):
        # Reshape data for neural networks
        if len(X.shape) == 4:
            X = X.reshape(X.shape[0], -1, X.shape[3])

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y.astype(int))
        self.augment = augment

        if normalize:
            self._normalize()

    def _normalize(self):
        normalized = torch.zeros_like(self.X)

        for i in range(self.X.shape[0]):
            for ch in range(self.X.shape[1]):
                data = self.X[i, ch, :]
                mean_val = torch.mean(data)
                std_val = torch.std(data)

                if std_val > 0:
                    normalized[i, ch, :] = (data - mean_val) / std_val
                else:
                    normalized[i, ch, :] = data

        self.X = normalized

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment:
            x = self._augment(x)

        return x, y

    def _augment(self, x):
        # Simple data augmentation
        if torch.rand(1) < 0.3:
            scale = torch.FloatTensor(1).uniform_(0.95, 1.05)
            x = x * scale

        if torch.rand(1) < 0.3:
            noise = torch.randn_like(x) * 0.01
            x = x + noise

        if torch.rand(1) < 0.3:
            shift = torch.randint(-2, 3, (1,)).item()
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)

        return x


def load_bci_data(data_path, subject_ids=None, use_filterbank=False):
    if subject_ids is None:
        subject_ids = list(range(1, 10))

    all_data = {}

    for subject_id in subject_ids:
        print(f"Loading subject {subject_id}...")

        try:
            # Load raw data
            dataset = BCIDataset(data_path, subject_id)
            raw = dataset.load_data()

            # Process data
            processor = EEGProcessor(use_filterbank=use_filterbank)
            raw = processor.preprocess(raw)
            epochs = processor.extract_epochs(raw)
            X, y = processor.extract_features(epochs)
            X = processor.normalize_data(X)

            all_data[f'S{subject_id:02d}'] = {
                'data': X,
                'labels': y,
                'n_trials': len(y),
                'n_bands': len(processor.filter_bands)
            }

            print(f"Subject {subject_id}: {len(y)} trials, shape {X.shape}")

        except Exception as e:
            print(f"Failed to load subject {subject_id}: {e}")
            continue

    return all_data


def create_dataloaders(X, y, batch_size=32, val_split=0.2, test_split=0.1, augment=True):
    # Split into train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )

    val_size = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )

    # Create datasets
    train_dataset = MotorImageryDataset(
        X_train, y_train, normalize=True, augment=augment)
    val_dataset = MotorImageryDataset(
        X_val, y_val, normalize=True, augment=False)
    test_dataset = MotorImageryDataset(
        X_test, y_test, normalize=True, augment=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    data_path = "data/BCICIV_2a_gdf"

    # Load data for one subject
    data = load_bci_data(data_path, subject_ids=[1], use_filterbank=True)

    if data:
        subject_data = data['S01']
        X = subject_data['data']
        y = subject_data['labels']

        print(f"\n--- Successfully loaded data:")
        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Number of filter bands: {subject_data['n_bands']}")

        train_loader, val_loader, test_loader = create_dataloaders(X, y, batch_size=16)

        print(f"\n--- Data loaders created:")
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")

        sample_batch = next(iter(train_loader))
        print(f"\nSample batch shape:")
        print(f"Data: {sample_batch[0].shape}")
        print(f"Labels: {sample_batch[1].shape}")

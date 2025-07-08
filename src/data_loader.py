import urllib.request
from pathlib import Path

from torch.utils.data import DataLoader
from torcheeg import model_selection, transforms
from torcheeg.datasets import BCICIV2aDataset


class MotorImageryDataLoader:
    def __init__(self,
                data_dir="./data",
                chunk_size=1750,
                num_channel=22,
                test_size=0.2,
                random_state=None,
                shuffle=False,
                num_worker=4):

        # Download if needed
        self.mat_folder = Path(data_dir) / "BCICIV_2a_mat"
        self._download_if_needed()

        # Load dataset
        self.dataset = BCICIV2aDataset(
            root_path=str(self.mat_folder),
            chunk_size=chunk_size,
            num_channel=num_channel,
            online_transform=transforms.Compose([
                transforms.To2d(),
                transforms.ToTensor()
            ]),
            label_transform=transforms.Compose([
                transforms.Select('label'),
                transforms.Lambda(lambda x: x - 1)
            ]),
            num_worker=num_worker
        )

        # Split dataset
        self.train_dataset, self.test_dataset = model_selection.train_test_split(
            dataset=self.dataset,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state
        )

    def _download_if_needed(self):
        base_url = "http://bnci-horizon-2020.eu/database/data-sets/001-2014/"
        files = ["A01T.mat", "A01E.mat", "A02T.mat", "A02E.mat", "A03T.mat", "A03E.mat",
                "A04T.mat", "A04E.mat", "A05T.mat", "A05E.mat", "A06T.mat", "A06E.mat",
                "A07T.mat", "A07E.mat", "A08T.mat", "A08E.mat", "A09T.mat", "A09E.mat"]

        self.mat_folder.mkdir(parents=True, exist_ok=True)

        missing_files = [f for f in files if not (self.mat_folder / f).exists()]
        if missing_files:
                print(f"BCICIV_2a dataset is missing. Downloading {len(missing_files)} files...")

        for filename in files:
            file_path = self.mat_folder / filename
            if not file_path.exists():
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(base_url + filename, file_path)

    def get_train_loader(self, batch_size=16, shuffle=True, num_workers=0):
        return DataLoader(self.train_dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    def get_test_loader(self, batch_size=16, shuffle=False, num_workers=0):
        return DataLoader(self.test_dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    def get_loaders(self, batch_size=16, shuffle_train=True, num_workers=0):
        return (self.get_train_loader(batch_size, shuffle_train, num_workers),
                self.get_test_loader(batch_size, False, num_workers))
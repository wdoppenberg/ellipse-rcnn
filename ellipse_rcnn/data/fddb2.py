import os
from pathlib import Path
from tempfile import TemporaryDirectory

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, extract_archive


class FDDB(Dataset):
    """
    Face Detection Data Set and Benchmark (FDDB) Dataset.
    
    Attributes
    ----------
    resources : dict
        Dictionary containing the URLs and their MD5 values for the dataset resources.
    
    Methods
    -------
    __init__(root, train, download, verbose)
        Initializes the dataset object and optionally downloads the data.
    _check_exists()
        Checks if the dataset has been downloaded and extracted properly.
    download()
        Downloads and extracts the FDDB dataset.
    __len__()
        Returns the number of entries in the dataset (to be implemented).
    __getitem__(index)
        Retrieves the entry at the specified index (to be implemented).
    
    Parameters
    ----------
    root : str or Path, optional
        Root directory of the dataset where ``FDDB/processed/training.pt`` and 
        ``FDDB/processed/test.pt`` exist. Defaults to './data/FDDB'.
    train : bool, optional
        If True, creates the dataset from ``training.pt``; otherwise, from ``test.pt``. 
        Defaults to True.
    download : bool, optional
        If True, downloads the dataset from the internet and stores it in the root directory.
        Defaults to False.
    verbose : bool, optional
        If True, enables verbose logging. Defaults to False.
    """

    resources = {
        "images": (
            "http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz",
            None,
        ),  # No MD5 provided
        "labels": (
            "http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz",
            None,
        ),  # No MD5 provided
    }

    def __init__(
            self,
            root: str | Path = Path("./data/FDDB"),
            train: bool = True,
            download: bool = False,
            verbose: bool = False
    ) -> None:
        """
        Initializes the FDDB dataset object.
    
        Parameters
        ----------
        root : str or Path, optional
            Root directory of the dataset where ``FDDB/processed/training.pt`` and 
            ``FDDB/processed/test.pt`` exist. Defaults to './data/FDDB'.
        train : bool, optional
            If True, creates the dataset from ``training.pt``; otherwise, from ``test.pt``. 
            Defaults to True.
        download : bool, optional
            If True, downloads the dataset from the internet and stores it in the root directory.
            Defaults to False.
        verbose : bool, optional
            If True, enables verbose logging. Defaults to False.
    
        Raises
        ------
        RuntimeError
            If the dataset is not found and `download` is False.
        """
        super().__init__()
        self.root: Path = Path(root)
        self.train = train
        self.verbose = verbose

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Use download=True to download it")

    def _check_exists(self) -> bool:
        """
        Check if the dataset has been downloaded and extracted properly.
    
        Returns
        -------
        bool
            True if the dataset is present, False otherwise.
        """
        images_path = self.root / "images"
        annotations_path = self.root / "labels"
        return images_path.exists() and annotations_path.exists()

    def download(self) -> None:
        """
        Download and extract the FDDB dataset.
    
        If the dataset already exists, no action is taken.
    
        Raises
        ------
        OSError
            If there is an issue during the download or extraction.
        """
        if self._check_exists():
            if self.verbose:
                print(f"FDDB Dataset already present under {self.root}.")
            return

        os.makedirs(self.root.absolute(), exist_ok=True)

        # Download and extract files
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for subfolder, (url, md5) in self.resources.items():
                if self.verbose:
                    print(f"Downloading from {url}")
                filename = os.path.basename(url)
                download_url(url, temp_path, filename, md5)
                extract_archive(temp_path / filename, self.root / subfolder)

        if self.verbose:
            print("Dataset downloaded and extracted successfully")

    def __len__(self) -> int:
        """
        Returns the number of entries in the dataset.
    
        Returns
        -------
        int
            The number of entries in the dataset.
        """
        raise NotImplementedError

    def __getitem__(self, index: int):
        """
        Retrieve the entry at the specified index.
    
        Parameters
        ----------
        index : int
            The index of the entry to retrieve.
    
        Returns
        -------
        Any
            The dataset entry corresponding to the specified index.
        """
        raise NotImplementedError


# Example usage
if __name__ == "__main__":
    # Create dataset instance
    dataset = FDDB(download=True, verbose=True)

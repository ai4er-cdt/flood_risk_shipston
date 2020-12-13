from typing import Any

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def rescale(self, data: Any, input: bool) -> Any:
        pass

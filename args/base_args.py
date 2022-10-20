
from dataclasses import dataclass

from utility.torch_utils import get_device


@dataclass
class BaseArguments:
    device: "torch.device"

    def __post_init__(self):
        self.device = get_device(self.device)

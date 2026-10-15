from .backbone import StructuredLayoutModel
from .framing import FramingModel
from .decoder import CoarseDecoder
from .config import validate_aes as validate


class Model(StructuredLayoutModel):
    def __init__(self, c, device='cuda:0'):
        validate(c)
        super().__init__(c, device)
        self.decoder = CoarseDecoder(c, self.language_dim).to(device)

    forward = FramingModel.forward

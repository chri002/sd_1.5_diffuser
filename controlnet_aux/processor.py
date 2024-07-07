"""
This file contains a Processor that can be used to process images with controlnet aux processors
"""
import io
import logging
from typing import Dict, Optional, Union
import sys
sys.path.append("../controlnet_aux/")

from PIL import Image
import cv2

from .canny import CannyDetector
from .shuffle import ContentShuffleDetector
from .leres import LeresDetector
from .normalbae import NormalBaeDetector
from .open_pose import OpenposeDetector

LOGGER = logging.getLogger(__name__)


MODELS = {
    # checkpoint models
    'openpose': {'class': OpenposeDetector, 'checkpoint': True},
    'openpose_face': {'class': OpenposeDetector, 'checkpoint': True},
    'openpose_faceonly': {'class': OpenposeDetector, 'checkpoint': True},
    'openpose_full': {'class': OpenposeDetector, 'checkpoint': True},
    'openpose_hand': {'class': OpenposeDetector, 'checkpoint': True},
    'normal_bae': {'class': NormalBaeDetector, 'checkpoint': True},
    'depth_leres': {'class': LeresDetector, 'checkpoint': True}, 
    'depth_leres++': {'class': LeresDetector, 'checkpoint': True}, 
    # instantiate
    'shuffle': {'class': ContentShuffleDetector, 'checkpoint': False},
    'canny': {'class': CannyDetector, 'checkpoint': False},
}


MODEL_PARAMS = {
    'openpose': {'include_body': True, 'include_hand': False, 'include_face': False},
    'openpose_face': {'include_body': True, 'include_hand': False, 'include_face': True},
    'openpose_faceonly': {'include_body': False, 'include_hand': False, 'include_face': True},
    'openpose_full': {'include_body': True, 'include_hand': True, 'include_face': True},
    'openpose_hand': {'include_body': False, 'include_hand': True, 'include_face': False},
    'normal_bae': {},
    'canny': {},
    'shuffle': {},
    'depth_zoe': {},
    'depth_leres': {'boost': False},
    'depth_leres++': {'boost': True},
}

CHOICES = f"Choices for the processor are {list(MODELS.keys())}"


class Processor:
    def __init__(self, processor_id: str, params: Optional[Dict] = None) -> None:
        """Processor that can be used to process images with controlnet aux processors

        Args:
            processor_id (str): processor name, options are 'hed, midas, mlsd, openpose,
                                pidinet, normalbae, lineart, lineart_coarse, lineart_anime,
                                canny, content_shuffle, zoe, mediapipe_face, tile'
            params (Optional[Dict]): parameters for the processor
        """
        LOGGER.info("Loading %s".format(processor_id))

        if processor_id not in MODELS:
            raise ValueError(f"{processor_id} is not a valid processor id. Please make sure to choose one of {', '.join(MODELS.keys())}")

        self.processor_id = processor_id
        self.processor = self.load_processor(self.processor_id)

        # load default params
        self.params = MODEL_PARAMS[self.processor_id]
        # update with user params
        if params:
            self.params.update(params)

    def load_processor(self, processor_id: str) -> 'Processor':
        """Load controlnet aux processors

        Args:
            processor_id (str): processor name

        Returns:
            Processor: controlnet aux processor
        """
        processor = MODELS[processor_id]['class']

        # check if the proecssor is a checkpoint model
        if MODELS[processor_id]['checkpoint']:
            processor = processor.from_pretrained("lllyasviel/Annotators")
        else:
            processor = processor()
        return processor

    def __call__(self, image: Union[Image.Image, bytes],
                 to_pil: bool = True, **kargs) -> Union[Image.Image, bytes]:
        """processes an image with a controlnet aux processor

        Args:
            image (Union[Image.Image, bytes]): input image in bytes or PIL Image
            to_pil (bool): whether to return bytes or PIL Image

        Returns:
            Union[Image.Image, bytes]: processed image in bytes or PIL Image
        """
        # check if bytes or PIL Image
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")

        processed_image = self.processor(image, **self.params, **kargs)

        if to_pil:
            return processed_image
        else:
            output_bytes = io.BytesIO()
            processed_image.save(output_bytes, format='JPEG')
            return output_bytes.getvalue()



if __name__ == "__main__":
    #processor_openPose = Processor("openpose_full")
    print("o")
    shuffle_model = Processor("shuffle")
    shuffle_model(cv2.imread("IMG_1.png", cv2.IMREAD_COLOR)).show()
    #processor_openPose(cv2.imread("IMG_1.png", cv2.IMREAD_COLOR), detect_resolution=1024).show()
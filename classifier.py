from typing import List

import imageio
import numpy as np
from bentoml import BentoService, api, artifacts, env
from bentoml.frameworks.keras import KerasModelArtifact
from bentoml.adapters import ImageInput
from bentoml.handlers import ImageHandler

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.applications import imagenet_utils


@env(requirements_txt_file="./requirements.txt")
@artifacts([KerasModelArtifact('classifier')])
class ImageClassifier(BentoService):
    def __init__(self):
        super().__init__()

    @api(
        input=ImageInput(),
        batch=False
    )
    def predict(self, input_image: imageio.core.util.Array) -> str:
        print(input_image.shape)

        plt.imshow(input_image)
        # plt.show()

        # img = image.smart_resize(input_image, (224, 224))
        img = image.array_to_img(input_image)
        img = img.resize((224, 224))

        #plt.imshow(img)
        #plt.show()

        final_image = np.expand_dims(img, axis=0) # need fourth dimension
        final_image = tf.keras.applications.mobilenet.preprocess_input(final_image)

        predictions = self.artifacts.classifier.predict(final_image)
        # print(predictions)

        result = imagenet_utils.decode_predictions(predictions)

        print(result)

        return result

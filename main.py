import tensorflow as tf
from bentoml_process import BentoML
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications import imagenet_utils


if __name__ == "__main__":
    # deep learning model weights - pre-trained
    mobile = tf.keras.applications.mobilenet.MobileNet()

    bento_ml = BentoML()
    bento_ml.run_bentoml(mobile)


    # img = image.load_img('./images/cat.jpeg', target_size=(224, 224))
    #
    # resized_img = image.img_to_array(img)
    # final_image = np.expand_dims(resized_img, axis=0)
    # final_image = tf.keras.applications.mobilenet.preprocess_input(final_image)
    #
    # predictions = mobile.predict(final_image)
    # print(predictions)
    #
    # results = imagenet_utils.decode_predictions(predictions)
    #
    # print(results)
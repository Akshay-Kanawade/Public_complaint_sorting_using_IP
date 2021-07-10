from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('models/public_model3.h5')


def test_on_img(img):
    data = []
    image = Image.open(img)
    image = image.resize((128, 128))
    data.append(np.array(image))
    x_test = np.array(data)
    y_pred = model.predict_classes(x_test)
    pred = model.predict(x_test)
    return image, y_pred, pred

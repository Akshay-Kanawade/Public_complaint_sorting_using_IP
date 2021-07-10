from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

model = load_model('/Users/KANAWADE AKSHAY/Desktop/project/pcm/models/public_model3.h5')

classes = {0: 'Electricity',
           1: 'Garbage',
           2: 'Potholes'
           }


def test_on_img(img):
    data = []
    image = Image.open(img)
    image = image.resize((128, 128))
    data.append(np.array(image))
    x_test = np.array(data)
    y_pred = model.predict_classes(x_test)
    pred = model.predict(x_test)
    return image, y_pred, pred


plot, prediction, pred = test_on_img(f"/Users/KANAWADE AKSHAY/Desktop/project/pcm/data/complaint_images/304.jpg")
print("pred", pred)
print("prediction", prediction)
s = [str(i) for i in prediction]
a = int("".join(s))
print("Predicted Image is: ", classes[a])
plt.imshow(plot)
plt.show()

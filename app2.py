
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing import image

model = keras.models.load_model("model.h5")
# model = tf.lite.TFLiteConverter.from_keras_model("model.h5")


def check(number):
    if number == 0:
        return 'Its not a Tumor'
    else:
        return 'Its a tumor'


def process_img(set_current, add_pixels_value=0):
    set_new = []
    for img in set_current:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Thresholding the image, and performing a series of erosions + dilations to remove any regions of noise
        thresh = cv2.threshold(img, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Finding the largest contour in the thresholded image
        (cnts, _) = cv2.findContours(thresh,
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("Contours: ", cnts)
        con = cv2.drawContours(img, cnts, -1, (0, 0, 255), 2)
        c = max(cnts, key=cv2.contourArea)

        # Finding the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        # Cropping the image using extreme points (left, right, top, bottom)
        new_img = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
        set_new.append(new_img)

    return np.array(set_new, dtype=object)


img = cv2.imread("../brain_tumor_dataset/yes/Y1.jpg")
# img = img.astype(np.float32)
cv2.imwrite("new/1.jpg", np.float32(img))

# img_array = image.img_to_array(
#     image.load_img("new/1.jpg", target_size=(224, 224)))
img_get = cv2.imread("new/1.jpg")
img_fin = cv2.resize(img_get, (224, 224))
img_array = np.array(img_fin)
img_array = process_img(img_array)
img_batch = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_batch)
prediction = [1 if x > 0.5 else 0 for x in prediction]
predicted_results = check(prediction)
print(predicted_results)

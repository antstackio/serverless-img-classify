import base64
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO

model_file = '/mnt/ml/SM-clf.h5'
model = tf.keras.models.load_model(model_file)
class_names = ['AC Unity', 'Hitman']

def lambda_handler(event, context):
    image_bytes = event['body'].encode('utf-8')
    image = Image.open(BytesIO(base64.b64decode(image_bytes)))
    image = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = 100 * np.max(tf.nn.softmax(predictions[0]))
    pred = class_names[np.argmax(predictions[0])]
    return {
        'statusCode': 200,
        'body': json.dumps(
            {
                "predicted_label": pred,
                "score": score,
            }
        )
    }
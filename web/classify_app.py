

import numpy as np
import time

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub
def main():


    classifier_model ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
    IMAGE_SHAPE = (224, 224)

    classifier = tf.keras.Sequential([
        hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
    ])
    grace_hopper = tf.keras.utils.get_file('image.jpg','image_file')
    grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
    grace_hopper
    grace_hopper = np.array(grace_hopper)/255.0
    grace_hopper.shape
    result = classifier.predict(grace_hopper[np.newaxis, ...])
    result.shape
    predicted_class = np.argmax(result[0], axis=-1)
    predicted_class
    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())
    plt.imshow(grace_hopper)
    plt.axis('off')
    predicted_class_name = imagenet_labels[predicted_class]
    retJson = {}
    retJson[0]=predicted_class_name
    # for node_id in top_k:
    #     human_string = node_lookup.id_to_string(node_id)
    #     score = predictions[node_id]
    #     retJson[human_string] = score.item()
    #     print('%s (score = %.5f)' % (human_string, score))
    print(retJson)
    with open("text.txt", 'w') as f:
        json.dump(retJson, f)
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_file',
        type=str,
        default='',
        help='Absolute path to image file.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

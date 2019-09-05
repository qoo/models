import numpy as np
import os
import sys
import tensorflow as tf

from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
print(tf.__version__)
# if tf.__version__ < '1.14.0':
#   raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
import collections
from tqdm import tqdm
import time


# ok:Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/root/train_out_finetune_forzen' + '/frozen_inference_graph.pb'
# !ls '../oid/train_out_finetune'
# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'oid_bbox_trainable_label_map.pbtxt')
MODEL_NAME = 'train_out_finetune_forzen_frozen_inference_graph'
# NUM_CLASSES = 545

### Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# read image list
def create_image_lists(image_dir):
    training_images = []
    if not tf.gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    result = collections.OrderedDict()
    sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        #     print(sub_dir)
        #     if is_root_dir:
        #       is_root_dir = False
        #       continue
        extensions = sorted(set(os.path.normcase(ext)  # Smash case on Windows.
                                for ext in ['JPEG', 'JPG', 'jpeg', 'jpg', 'png']))
        file_list = []
        dir_name = image_dir
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, '*.' + extension)
            file_list.extend(tf.gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        if len(file_list) < 20:
            tf.logging.warning(
                'WARNING: Folder has less than 20 images, which may cause issues.')

        #     label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())

        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)

            training_images.append(base_name)

    return training_images


def create_bottleneck_file(image_path, bottleneck_path, sess, bottleneck_tensor, image_tensor):
    if not os.path.exists(bottleneck_path):
        bottleneck_values = create_bottleneck_value(image_path, sess, bottleneck_tensor, image_tensor)
        if not os.path.exists('/root/bottleneck.shape.npy'):
            np.save('/root/bottleneck.shape.npy', bottleneck_values.shape)
        bottleneck_values = bottleneck_values.flatten()
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with tf.gfile.GFile(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)


def create_bottleneck_value(image_path, sess, bottleneck_tensor, image_tensor):
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Run inference
    bottleneck_values = sess.run(bottleneck_tensor,
                                 feed_dict={image_tensor: np.expand_dims(image_np, 0)})
    return bottleneck_values



# image_dir = '/root/images'
image_dir = '/root/raw_images_train'

bottleneck_dir = '/root/bottleneck'
error_dir = '/root'
image_lists = create_image_lists(image_dir)
# print(image_lists)


# errorSet = set()
with detection_graph.as_default():
    sess = tf.Session()
    ops = tf.get_default_graph().get_operations()

    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    bottleneck_tensor = tf.get_default_graph().get_tensor_by_name('SecondStageBoxPredictor/AvgPool:0')
    with tqdm(total=len(image_lists)) as pbar:
        for step, image_name in enumerate(image_lists):
            pbar.update(step)
            file_name, ext = image_name.split('.')
            image_path = os.path.join(image_dir, image_name)
            bottleneck_path = os.path.join(bottleneck_dir, file_name)
            #             print(image_path, bottleneck_path)
            loop_start = time.time()
            try:
                create_bottleneck_file(image_path, bottleneck_path, sess, bottleneck_tensor, image_tensor)
            except:
                tf.logging.warning('Fail to create bottleneck: ' + image_name)
                #                 errorSet.add(image_name)
                with open(os.path.join(error_dir, 'error.txt'), 'a') as f:
                    f.write("%s\n" % str(image_name))
            lfw_time = time.time() - loop_start
            print('Total Inference time: {} sec @ {} image'.format(lfw_time, image_name))

    # Run inference
#     output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image_np, 0)})





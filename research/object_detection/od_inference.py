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
import argparse
import pickle
import shutil




def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str,
        help='Path to the data directory containing images patches.')
    parser.add_argument('--model', type=str,
        help='Could be either a directory containing a model protobuf (.pb) file')
    parser.add_argument('--image_resize', type=int,
        help='Image size (height, width) in pixels.', default=600)
    parser.add_argument('--inference_out', type=str,
                        help='The path to save the result, if not specify, '' ', default='~/xx')
    return parser.parse_args(argv)




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


def create_inference_file(image_path, bottleneck_path, sess, tensor_dict, image_tensor, image_resize):
    if not os.path.exists(bottleneck_path):
        ###  create tempfile
        with open(bottleneck_path, 'wb') as handle:
            pickle.dump(image_path, handle, protocol=pickle.HIGHEST_PROTOCOL)
        output_dict = inference_single_image(image_path, sess, tensor_dict, image_tensor, image_resize)
        # if not os.path.exists('/root/bottleneck.shape.npy'):
        #     np.save('/root/bottleneck.shape.npy', bottleneck_values.shape)
        with open(bottleneck_path, 'wb') as handle:
            pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # bottleneck_values = bottleneck_values.flatten()
        # bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        # with tf.gfile.GFile(bottleneck_path, 'w') as bottleneck_file:
        #     bottleneck_file.write(bottleneck_string)
def inference_single_image(image_path, sess_, tensor_dict, image_tensor, image_resize):
    image = Image.open(image_path)
    image = image.resize((image_resize, image_resize))
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    output_dict = sess_.run(tensor_dict,
                     feed_dict={image_tensor: np.expand_dims(image_np, 0)})
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


args = parse_arguments(sys.argv[1:])


# ok:Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = '/root/train_out_finetune_forzen' + '/frozen_inference_graph.pb'
PATH_TO_CKPT=args.model
print('Loading model: {}'.format(PATH_TO_CKPT))
# !ls '../oid/train_out_finetune'
# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'oid_bbox_trainable_label_map.pbtxt')
# MODEL_NAME = 'train_out_finetune_forzen_frozen_inference_graph'
# NUM_CLASSES = 545


### Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# image_dir = '/root/images'
# image_dir = '/root/raw_images_train'
image_dir = args.image_dir

# bottleneck_dir = '/root/bottleneck'
bottleneck_dir = args.inference_out

error_dir = bottleneck_dir + '_errorLog'
image_lists = create_image_lists(image_dir)

image_resize = args.image_resize
# print(image_lists)
if not os.path.exists(error_dir):
    os.makedirs(error_dir)

if not os.path.exists(bottleneck_dir):
    os.makedirs(bottleneck_dir)

# errorSet = set()
with detection_graph.as_default():
    sess = tf.Session()
    ops = tf.get_default_graph().get_operations()

    # image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    # bottleneck_tensor = tf.get_default_graph().get_tensor_by_name('SecondStageBoxPredictor/AvgPool:0')

    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)
    # if 'detection_masks' in tensor_dict:
    #     # The following processing is only for single image
    #     detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
    #     detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
    #     # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
    #     real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
    #     detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
    #     detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
    #     detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
    #         detection_masks, detection_boxes, image.shape[0], image.shape[1])
    #     detection_masks_reframed = tf.cast(
    #         tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    #     # Follow the convention by adding back the batch dimension
    #     tensor_dict['detection_masks'] = tf.expand_dims(
    #         detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')


    for step, image_name in enumerate(image_lists):

        file_name, ext = image_name.split('.')
        image_path = os.path.join(image_dir, image_name)
        bottleneck_path = os.path.join(bottleneck_dir, file_name+'.pickle')
        #             print(image_path, bottleneck_path)
        loop_start = time.time()
        try:
            create_inference_file(image_path, bottleneck_path, sess, tensor_dict, image_tensor, image_resize)
        except:
            tf.logging.warning('Fail to create bottleneck: ' + image_name)
            #                 errorSet.add(image_name)
            with open(os.path.join(error_dir, 'error.txt'), 'a') as f:
                f.write("%s\n" % str(image_name))
        lfw_time = time.time() - loop_start
        print("{}/{}: {} sec. {}".format(step, len(image_lists), lfw_time, image_name))
        with open(os.path.join(error_dir, 'log.txt'), 'a') as f:
            f.write('Total Inference time: {} sec @ {} image\n'.format(lfw_time, image_name))
        # print('Total Inference time: {} sec @ {} image'.format(lfw_time, image_name))


### example od_inference.py
# python aa.py --image_dir --model --image_resize --inference_out
# python aa.py --image_dir ~/1252812627.ori.1sec \
# --model ~/oid_lrRate_per300_ckpt800000/train_out_547_1GPU1batch_lrPet_frozen_800000/frozen_inference_graph.pb \
# --image_resize 600 \
# --inference_out ~/od_infrence_1252812627


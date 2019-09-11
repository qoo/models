"""Example usage:
  python object_detection/dataset_tools/create_oid_tf_record.py \
    --input_annotations_csv=/path/to/input/annotations-human-bbox.csv \
    --input_images_directory=/path/to/input/image_pixels_directory \
    --input_label_map=/path/to/input/labels_bbox_545.labelmap \
    --output_tf_record_path_prefix=/path/to/output/prefix.tfrecord
CSVs with bounding box annotations and image metadata (including the image URLs)
can be downloaded from the Open Images GitHub repository:
https://github.com/openimages/dataset
This script will include every image found in the input_images_directory in the
output TFRecord, even if the image has no corresponding bounding box annotations
in the input_annotations_csv.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import contextlib2
import pandas as pd
import tensorflow as tf

from object_detection.dataset_tools import oid_tfrecord_creation
from object_detection.utils import label_map_util
import json
import sys



print ("This is the name of the script: ", sys.argv[0])
print ("Number of arguments: ", len(sys.argv))
print ("The arguments are: " , str(sys.argv))

metric_dir = sys.argv[1]
# metric_output = sys.argv[2]


df = pd.read_csv(os.path.join(metric_dir ,'metrics.csv'), names=['class', 'AP@0.5'])

filtered_df = df[df['AP@0.5'].notnull()]['AP@0.5']
mAP_dict = {}
mAP_dict['mAP'] = filtered_df.mean()
query_df = df[df['class'].str.contains('Coconut', regex=False)]
if len(query_df) > 0:
    mAP_dict['Coconut'] = query_df.iloc[0]['AP@0.5']
else:
    mAP_dict['Coconut'] = 'NOT EXIST'
query_df = df[df['class'].str.contains('Parking meter', regex=False)]
if len(query_df) > 0:
    mAP_dict['Parking meter'] = query_df.iloc[0]['AP@0.5']
else:
    mAP_dict['Parking meter'] = 'NOT EXIST'
mAP_dict['metric_path'] = os.path.join(metric_dir ,'metrics.csv')

##display
print("Eval for {}".format(mAP_dict['metric_path']))
print("mAP@0.5: {}".format(mAP_dict['mAP']))
print("Coconut AP@0.5: {}".format(mAP_dict['Coconut']))
print("Parking meter AP@0.5: {}".format(mAP_dict['Parking meter']))
print("Saving to: {}".format(os.path.join(metric_dir ,'metrics.json')))

json = json.dumps(mAP_dict)
f = open(os.path.join(metric_dir ,'metrics.json'),"w")
f.write(json)
f.close()

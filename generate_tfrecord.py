"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python3 generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record
  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
   if row_label == 'GD1' or row_label == '11':
   	return 1
   elif row_label == 'GD2' or row_label == '12':
   	return 2
   elif row_label == 'GD3' or row_label == '13':
   	return 3
   elif row_label == 'GD4' or row_label == '14':
   	return 4
   elif row_label == 'GD5' or row_label == '15':
   	return 5
   elif row_label == 'GD6' or row_label == '16':
   	return 6
   elif row_label == 'GD7' or row_label == '17':
   	return 7
   elif row_label == 'GD8' or row_label == '18':
   	return 8
   elif row_label == 'DD1' or row_label == '41':
   	return 9
   elif row_label == 'DD2' or row_label == '42':
   	return 10
   elif row_label == 'DD3' or row_label == '43':
   	return 11
   elif row_label == 'DD4' or row_label == '44':
   	return 12
   elif row_label == 'DD5' or row_label == '45':
   	return 13
   elif row_label == 'DD6' or row_label == '46':
   	return 14
   elif row_label == 'DD7' or row_label == '47':
   	return 15
   elif row_label == 'DD8' or row_label == '48':
   	return 16
   elif row_label == 'GL1' or row_label == '21':
   	return 17
   elif row_label == 'GL2' or row_label == '22':
   	return 18
   elif row_label == 'GL3' or row_label == '23':
   	return 19
   elif row_label == 'GL4' or row_label == '24':
   	return 20
   elif row_label == 'GL5' or row_label == '25':
   	return 21
   elif row_label == 'GL6' or row_label == '26':
   	return 22
   elif row_label == 'GL7' or row_label == '27':
   	return 23
   elif row_label == 'GL8' or row_label == '28':
   	return 24
   elif row_label == 'DL1' or row_label == '31':
   	return 25
   elif row_label == 'DL2' or row_label == '32':
   	return 26
   elif row_label == 'DL3' or row_label == '33':
   	return 27
   elif row_label == 'DL4' or row_label == '34':
   	return 28
   elif row_label == 'DL5' or row_label == '35':
   	return 29
   elif row_label == 'DL6' or row_label == '36':
   	return 30
   elif row_label == 'DL7' or row_label == '37':
   	return 31
   elif row_label == 'DL8' or row_label == '38':
        return 32
   else : 
        return 33

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), 'images')
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()

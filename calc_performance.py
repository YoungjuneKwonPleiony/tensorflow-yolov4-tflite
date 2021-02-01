import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('data', './', 'path of data')
flags.DEFINE_string('output', './output', 'path of data')
flags.DEFINE_string('image_ext', '.png', 'extention of images')
flags.DEFINE_boolean('segmented_data', True, 'segmented or not')
flags.DEFINE_boolean('save_output', True, 'save output or not')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

class Detector:
  def __init__(self, flags):
    self.config = ConfigProto()
    self.config.gpu_options.allow_growth = True
    self.session = InteractiveSession(config=self.config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(flags)
    self.input_size = flags.size
    self.data_path = flags.data
    self.flags = flags
    self.saved_model_loaded = tf.saved_model.load(self.flags.weights, tags=[tag_constants.SERVING])
    self.init_colors()

  def find_files(self, suffix):
    import os
    pys = []
    for p, d, f in os.walk(self.data_path):
      for file in f:
        pys.append(f'{p}/{file}') if file.endswith(suffix) else None
    return pys

  def cal_bbox(box):
      dw = 1./(640)
      dh = 1./(480)
      return [box[1] * dh, box[0] * dw, box[3] * dh, box[2] * dw]

  def extract_answers_for(self, image):
    annotation_file = f'{image[:-6]}.txt' if self.flags.segmented_data else f'{image[:-3]}.txt'
    with open(annotation_file, 'r') as f:
      result = [[float(f) for f in l.split()] for l in f.read().strip().split('\n')]
    return [Detector.cal_bbox(b) for b in result] if self.flags.segmented_data else result

  def load_image(self, image):
    return cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

  def bb_iou(a, b):
    # [t, l, b, r]
    xA = max(a[1], b[1])
    yA = max(a[0], b[0])
    xB = min(a[3], b[3])
    yB = min(a[2], b[2])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (a[2] - a[0]) * (a[3] - a[1])
    boxBArea = (b[2] - b[0]) * (b[3] - b[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

  def process_for_np_image(self, np_image):
    image_data = cv2.resize(np_image, (self.input_size, self.input_size))
    images_data = np.asarray([image_data / 255.]).astype(np.float32)

    pred_bbox = self.saved_model_loaded.signatures['serving_default'](tf.constant(images_data))
    for key, value in pred_bbox.items():
      boxes = value[:, :, 0:4]
      pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
      boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
      scores=tf.reshape(
          pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
      max_output_size_per_class=50,
      max_total_size=50,
      iou_threshold=FLAGS.iou,
      score_threshold=FLAGS.score
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    return pred_bbox

  def evaluate_performance(self):
    for i in self.find_files(self.flags.image_ext):
      ans = self.extract_answers_for(i)

  def init_colors(self, num_classes=50):
    import colorsys
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    clrs = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), clrs))

  def calc_ious(self, answers, bboxs):
    import itertools
    return [e for e in [(Detector.bb_iou(*p), p) for p in itertools.product(answers, bboxs[0]) if sum(p[0]) > 0 and sum(p[1])] if e[0] > 0.1]

  def draw_rect(self, np_image, bbox, color_idx=0):
    image_h, image_w, _ = np_image.shape
    coor = [
      int(bbox[0] * image_h),
      int(bbox[1] * image_w),
      int(bbox[2] * image_h),
      int(bbox[3] * image_w)
    ]
    bbox_thick = int(0.6 * (image_h + image_w) / 600)
    c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
    cv2.rectangle(np_image, c1, c2, self.colors[color_idx], bbox_thick)

  def save_output(self, output, np_image, ious):
    [self.draw_rect(np_image, i[1][0], color_idx=10) for i in ious]
    [self.draw_rect(np_image, i[1][1]) for i in ious]
    img = Image.fromarray(np_image.astype(np.uint8))
    cv2.imwrite(output, cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))

  def evaluate_for_image(self, image):
    import os
    answers = self.extract_answers_for(image)
    np_image = self.load_image(image)
    pred_bbox = self.process_for_np_image(np_image)
    ious = self.calc_ious(answers, pred_bbox[0])
    output = f'{self.flags.output}/images/{image.replace(self.data_path, "").replace(os.path.sep, "_")}'
    self.save_output(output, np_image, ious) if self.flags.save_output else None
    return [i[0] for i in ious], len(answers)

def main(_argv):
  import pandas
  detector = Detector(FLAGS)
  files = detector.find_files('.png')
  results = []
  for i, f in enumerate(files):
    print(f'{i} / {len(files)}')
    e = detector.evaluate_for_image(f)
    results.append({
      'file': f,
      'avg_iou': sum(e[0]) / e[1],
      'cnt_iou': len(e[0]),
      'ans': e[1],
      'loss': e[1] - len(e[0])
    })
  pandas.DataFrame(results).to_csv(f'{FLAGS.output}/report.csv')

if __name__ == '__main__':
  try:
    app.run(main)
  except SystemExit:
    pass

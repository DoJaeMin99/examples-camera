import argparse
import time
import os

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import requests
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter



def draw_objects(draw, objs, labels, line_width=5):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                       outline='green', width=line_width)
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                  '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                  fill='green')


def main():
    default_model_dir = os.environ['HOME']
    default_model_dir = os.path.join(default_model_dir, 'google-coral/examples-camera/all_models')
    default_model = 'Coregleam_hotspot_model_500_v2.tflite'
    default_labels = 'Coregleam_panel_labels.txt'
    folder_path = '/home/mendel/google-coral/examples-camera/image/hotspot_image'
    index = 0
    for image in os.listdir(folder_path):
        output = '/home/mendel/google-coral/examples-camera/result_image/hotspot_result/hotspot_{}.jpg'.format(index)
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--model', default=os.path.join(default_model_dir, default_model))
        parser.add_argument('--input', default=os.path.join(folder_path, image))
        parser.add_argument('--labels', default=os.path.join(default_model_dir, default_labels))
        parser.add_argument('-t', '--threshold', type=float, default=0.4,
                            help='Score threshold for detected objects')
        parser.add_argument('--output', default=os.path.join(default_model_dir, output))
        index += 1
        parser.add_argument('-c', '--count', type=int, default=5,
                            help='Number of times to run inference')
        args = parser.parse_args()

        labels = read_label_file(args.labels) if args.labels else {}
        interpreter = make_interpreter(args.model)
        interpreter.allocate_tensors()

        image = Image.open(args.input)
        _, scale = common.set_resized_input(
            interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

        print('----INFERENCE TIME----')
        print('Note: The first inference is slow because it includes',
            'loading the model into Edge TPU memory.')
        for _ in range(args.count):
            start = time.perf_counter()
            interpreter.invoke()
            inference_time = time.perf_counter() - start
            objs = detect.get_objects(interpreter, args.threshold, scale)
            print('%.2f ms' % (inference_time * 1000))

        print('-------RESULTS--------')
        if not objs:
            print('No objects detected')
        else:
            for obj in objs:
                print(labels.get(obj.id, obj.id))
                print('  id:    ', obj.id)
                print('  score: ', obj.score)
                print('  bbox:  ', obj.bbox)

        if args.output:
            image = image.convert('RGB')
            draw_objects(ImageDraw.Draw(image), objs, labels)
            image.save(args.output)
            image.show()


if __name__ == '__main__':
    main()

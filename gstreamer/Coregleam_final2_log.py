import argparse
from itertools import count
import gstreamer
import os
import time
import csv

from common import avg_fps_counter, SVG
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

import socket
import threading

udp_file_name = ' '
decoded_file_name = ''
csv_save_dir = '/home/mendel/google-coral/examples-camera/log'
cnt = 1
last_saved_file_name = ''  # Variable to track the last saved file name
time_to_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
############# UDP Server thread func
def Server_task():
    global udp_file_name, decoded_file_name
    
    server_ip = '192.168.0.188'
    server_port = 7942

    # UDP 서버 소켓 생성
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 소켓과 IP 주소, 포트를 바인딩
    server_socket.bind((server_ip, server_port))
    print("Start UDP server.")

    while True:
        # 데이터 수신
        udp_file_name, _ = server_socket.recvfrom(1024)
        print("Data from client:", udp_file_name.decode())
        decoded_file_name = udp_file_name.decode()
        time.sleep(0.0001)
#############

def save_csv(text_lines):
    global csv_save_dir, last_saved_file_name , time_to_time, cnt

    csv_file_name = os.path.join(csv_save_dir, f'waring_{time_to_time}.csv')

    # Extract GT and Detect values
    gt = None
    detect = None
    for line in text_lines:
        if line.startswith('GT'):
            gt = int(line.split(':')[1].strip())
        elif line.startswith('DETECT'):
            detect = int(line.split(':')[1].strip())

    # Calculate accuracy if GT and Detect values are available
    accuracy = ''
    if gt is not None and detect is not None:
        accuracy = 'Accuracy: {:.2f}%'.format(detect / gt * 100)

    # Check if the last saved file name is the same as the current file name
    if last_saved_file_name != decoded_file_name:
        with open(csv_file_name, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(text_lines + [accuracy])
            last_saved_file_name = decoded_file_name


def generate_svg(src_size, inference_box, objs, labels, text_lines):
    svg = SVG(src_size)
    src_w, src_h = src_size
    box_x, box_y, box_w, box_h = inference_box
    scale_x, scale_y = src_w / box_w, src_h / box_h

    for y, line in enumerate(text_lines, start=1):
        svg.add_text(10, y * 20, line, 20)
    for obj in objs:
        bbox = obj.bbox
        if not bbox.valid:
            continue
        # Absolute coordinates, input tensor space.
        x, y = bbox.xmin, bbox.ymin
        w, h = bbox.width, bbox.height
        # Subtract boxing offset.
        x, y = x - box_x, y - box_y
        # Scale to source coordinate space.
        x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
        percent = int(100 * obj.score)
        if percent < 48:
            continue
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
        svg.add_text(x, y - 5, label, 20)
        svg.add_rect(x, y, w, h, 'green', 3)
    return svg


def main_test():
    thread = threading.Thread(target=Server_task)
    thread.start()

def main():
    default_model_dir = '../all_models'
    default_model = 'TTA_waring_model_500.tflite'
    default_labels = 'Coregleam_waring_label.txt'

    thread = threading.Thread(target=Server_task)
    thread.start()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--videosrc', help='Which video source to use. ',
                        default='/dev/video1')
    parser.add_argument('--videofmt', help='Input video format.',
                        default='raw',
                        choices=['raw', 'h264', 'jpeg'])
    parser.add_argument('--box', choices=['on', 'off'], default='on',
                        help='Whether to draw rectangles around objects')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    # Average fps over last 30 frames.
    fps_counter = avg_fps_counter(30)

    def user_callback(input_tensor, src_size, inference_box):
        global udp_file_name, decoded_file_name
        nonlocal fps_counter

        start_time = time.monotonic()
        run_inference(interpreter, input_tensor)
        objs = get_objects(interpreter, args.threshold)[:args.top_k]
        box_count = len(objs)
        end_time = time.monotonic()

        div = decoded_file_name.split(' ')
        text_lines = [
            'Inference: {:.2f} ms'.format((end_time - start_time) * 1000),
            'FPS: {} fps'.format(round(next(fps_counter))),
            '',
            'name : {}'.format(div[0]),
            'GT : {}'.format(div[1]),
            'DETECT : {}'.format(box_count)
        ]

        if args.box == 'on':
            svg = generate_svg(src_size, inference_box, objs, labels, text_lines)
        else:
            svg = SVG(src_size)
            for y, line in enumerate(text_lines, start=1):
                svg.add_text(10, y * 20, line, 20)
        
        text_lines_2 = [
            'name : {}'.format(div[0]),
            '{}'.format(div[1]),
            '{}'.format(box_count)
        ]
        save_csv(text_lines_2)  # Save text_lines to CSV file
        return svg.finish()

    result = gstreamer.run_pipeline(user_callback,
                                    src_size=(640, 480),
                                    appsink_size=inference_size,
                                    videosrc=args.videosrc,
                                    videofmt=args.videofmt)

if __name__ == '__main__':
    main()

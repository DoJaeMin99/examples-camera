import argparse
import gstreamer

def process_video(data):
    pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--videosrc', help='Which video source to use. ',
                        default='/dev/video1')
    parser.add_argument('--videofmt', help='Input video format.',
                        default='raw',
                        choices=['raw', 'h264', 'jpeg'])
    args = parser.parse_args()


    result = gstreamer.run_pipeline(src_size=(640, 480),
                                    appsink_size=(300, 300),
                                    videosrc=args.videosrc,
                                    videofmt=args.videofmt,
                                    user_function=process_video)

if __name__ == '__main__':
    main()

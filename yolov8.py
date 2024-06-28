import os
import cv2
import sys
import argparse
import time
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('SFT')+1]))

from py_utils.coco_utils import COCO_test_helper
import numpy as np


OBJ_THRESH = 0.25
NMS_THRESH = 0.45

# The follew two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


class PerformanceMonitor:
    def __init__(self):
        self.timings = {
            'read': [],
            'preprocess': [],
            'inference': [],
            'postprocess': [],
            'draw': [],
            'display': []
        }
    
    def add_timing(self, stage, time):
        self.timings[stage].append(time)
    
    def get_average_timings(self):
        return {stage: np.mean(times) if times else 0 for stage, times in self.timings.items()}


class VideoProcessor:
    def __init__(self, model, co_helper, video_path=None, use_webcam=False, queue_size=1):
        self.model = model
        self.co_helper = co_helper
        self.video_path = video_path
        self.use_webcam = use_webcam
        self.frame_queue = Queue(maxsize=queue_size)
        self.result_queue = Queue(maxsize=queue_size)
        self.is_running = False
        self.total_frames = 0
        self.total_time = 0
        self.display_fps = 0
        self.performance_monitor = PerformanceMonitor()

    def start(self):
        self.is_running = True
        threading.Thread(target=self._read_frames).start()
        threading.Thread(target=self._process_frames).start()
        threading.Thread(target=self._display_results).start()

    def stop(self):
        self.is_running = False

    def _read_frames(self):
        cap = cv2.VideoCapture(0 if self.use_webcam else self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open {'webcam' if self.use_webcam else 'video file ' + self.video_path}")
            self.stop()
            return

        while self.is_running:
            if not self.frame_queue.full():
                ret, frame = cap.read()
                if not ret:
                    self.stop()
                    break
                self.frame_queue.put(frame)
            else:
                time.sleep(0.0001)  # Short sleep to prevent CPU overuse
        cap.release()

    def _process_frames(self):
        while self.is_running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                start_time = time.time()
                img = self.co_helper.letter_box(im=frame, new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                input_data = np.expand_dims(img, axis=0)
                self.performance_monitor.add_timing('preprocess', time.time() - start_time)

                start_time = time.time()
                outputs = self.model.run([input_data])
                inference_time = time.time() - start_time
                self.performance_monitor.add_timing('inference', inference_time)
                self.total_time += inference_time
                self.total_frames += 1

                start_time = time.time()
                if outputs is not None:
                    boxes, classes, scores = post_process(outputs)
                    if boxes is not None:
                        frame = draw(frame, self.co_helper.get_real_box(boxes), scores, classes)
                self.performance_monitor.add_timing('postprocess', time.time() - start_time)
                
                fps = 1.0 / inference_time if inference_time > 0 else 0
                self.display_fps = fps  # Update FPS for display
                
                if not self.result_queue.full():
                    self.result_queue.put(frame)
            else:
                time.sleep(0.0001)  # Short sleep to prevent CPU overuse

    def _display_results(self):
        last_time = time.time()
        frame_count = 0
        while self.is_running:
            if not self.result_queue.empty():
                frame = self.result_queue.get()
                current_time = time.time()
                frame_count += 1
                if current_time - last_time >= 1.0:  # Update FPS every second
                    self.display_fps = frame_count / (current_time - last_time)
                    frame_count = 0
                    last_time = current_time

                cv2.putText(frame, f"FPS: {self.display_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("YOLOv8 RKNN Video", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
            else:
                time.sleep(0.0001)  # Short sleep to prevent CPU overuse

    def run(self):
        self.start()
        while self.is_running:
            time.sleep(0.001)  # Small delay to prevent busy-waiting
        cv2.destroyAllWindows()
        avg_fps = self.total_frames / self.total_time if self.total_time > 0 else 0
        print(f"Video processing complete. Frames processed: {self.total_frames}, Average FPS: {avg_fps:.2f}")
        
        # Print detailed performance metrics
        avg_timings = self.performance_monitor.get_average_timings()
        print("\nPerformance Breakdown:")
        for stage, avg_time in avg_timings.items():
            print(f"{stage.capitalize()}: {avg_time*1000:.2f} ms")
        
        total_avg_time = sum(avg_timings.values())
        print(f"\nTotal average processing time per frame: {total_avg_time*1000:.2f} ms")
        if total_avg_time > 0:
            print(f"Theoretical max FPS: {1/total_avg_time:.2f}")
        else:
            print("No frames were processed, theoretical max FPS cannot be calculated.")


def run_video(model, co_helper, video_path):
    processor = VideoProcessor(model, co_helper, video_path, use_webcam=False, queue_size=1)
    processor.run()


def web_cam(model, co_helper):
    processor = VideoProcessor(model, co_helper, use_webcam=True, queue_size=1)
    processor.run()


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis = axis, keepdims=True)

def dfl(position):
    # Distribution Focal Loss (DFL)
    n,c,h,w = position.shape
    p_num = 4
    mc = c//p_num
    y = position.reshape(n,p_num,mc,h,w)
    y = softmax(y,2)
    acc_metrix = np.array(range(mc),dtype=float).reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    for i, (box, score, cl) in enumerate(zip(boxes, scores, classes)):
        try:
            top, left, right, bottom = [int(_b) for _b in box]
            print(f"Drawing box {i}: {CLASSES[cl]} @ ({top} {left} {right} {bottom}) {score:.3f}")
            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, f'{CLASSES[cl]} {score:.2f}',
                        (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        except Exception as e:
            print(f"Error drawing box {i}: {e}")
    return image


def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
        platform = 'pytorch'
        from py_utils.pytorch_executor import Torch_model_container
        model = Torch_model_container(args.model_path)
    elif model_path.endswith('.rknn'):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container 
        model = RKNN_model_container(args.model_path, args.target, args.device_id)
    elif model_path.endswith('onnx'):
        platform = 'onnx'
        from py_utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(args.model_path)
    else:
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform

def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # basic params
    parser.add_argument('--model_path', type=str, required= True, help='model path, could be .pt or .rknn file')
    parser.add_argument('--target', type=str, default='rk3588', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')
    
    parser.add_argument('--img_show', action='store_true', default=False, help='draw the result and show')
    parser.add_argument('--img_save', action='store_true', default=False, help='save the result')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--web_cam', action='store_true', help='Use the webcam for input')




    # data params
    parser.add_argument('--anno_json', type=str, default='../../../datasets/COCO/annotations/instances_val2017.json', help='coco annotation path')
    # coco val folder: '../../../datasets/COCO//val2017'
    parser.add_argument('--img_folder', type=str, default='./model', help='img folder path')
    parser.add_argument('--coco_map_test', action='store_true', help='enable coco map test')

    args = parser.parse_args()

    # init model
    model, platform = setup_model(args)
    co_helper = COCO_test_helper(enable_letter_box=True)

    file_list = sorted(os.listdir(args.img_folder))
    img_list = []
    for path in file_list:
        if img_check(path):
            img_list.append(path)
    co_helper = COCO_test_helper(enable_letter_box=True)

    if args.web_cam:
         web_cam(model, co_helper)
    elif args.video:
        run_video(model, co_helper, args.video)
    else:
        # run test
        for i in range(len(img_list)):
            print('infer {}/{}'.format(i+1, len(img_list)), end='\r')

            img_name = img_list[i]
            img_path = os.path.join(args.img_folder, img_name)
            if not os.path.exists(img_path):
                print("{} is not found", img_name)
                continue

            img_src = cv2.imread(img_path)
            if img_src is None:
                continue

            '''
            # using for test input dumped by C.demo
            img_src = np.fromfile('./input_b/demo_c_input_hwc_rgb.txt', dtype=np.uint8).reshape(640,640,3)
            img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)
            '''

            # Due to rga init with (0,0,0), we using pad_color (0,0,0) instead of (114, 114, 114)
            pad_color = (0,0,0)
            img = co_helper.letter_box(im= img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # preprocee if not rknn model
            if platform in ['pytorch', 'onnx']:
                input_data = img.transpose((2,0,1))
                input_data = input_data.reshape(1,*input_data.shape).astype(np.float32)
                input_data = input_data/255.
            else:
                input_data = img

            input_data = np.expand_dims(img, axis=0)
            outputs = model.run([input_data])
            boxes, classes, scores = post_process(outputs)

            if args.img_show or args.img_save:
                print('\n\nIMG: {}'.format(img_name))
                img_p = img_src.copy()
                if boxes is not None:
                    draw(img_p, co_helper.get_real_box(boxes), scores, classes)

                if args.img_save:
                    if not os.path.exists('./result'):
                        os.mkdir('./result')
                    result_path = os.path.join('./result', img_name)
                    cv2.imwrite(result_path, img_p)
                    print('Detection result save to {}'.format(result_path))
                            
                if args.img_show:
                    cv2.imshow("full post process result", img_p)
                    cv2.waitKeyEx(0)

            # record maps
            if args.coco_map_test is True:
                if boxes is not None:
                    for i in range(boxes.shape[0]):
                        co_helper.add_single_record(image_id = int(img_name.split('.')[0]),
                                                    category_id = coco_id_list[int(classes[i])],
                                                    bbox = boxes[i],
                                                    score = round(scores[i], 5).astype(np.float)
                                                    )
    # calculate maps
    if args.coco_map_test is True:
        pred_json = args.model_path.split('.')[-2]+ '_{}'.format(platform) +'.json'
        pred_json = pred_json.split('/')[-1]
        pred_json = os.path.join('./', pred_json)
        co_helper.export_to_json(pred_json)

        from py_utils.coco_utils import coco_eval_with_json
        coco_eval_with_json(args.anno_json, pred_json)
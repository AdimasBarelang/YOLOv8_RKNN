from queue import Queue
from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor, as_completed

class RKNN_model_container:
    def __init__(self, model_path, target=None, device_id=None) -> None:
        self.rknn = RKNNLite()
        
        # Load RKNN Model
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            print("Load RKNN model failed")
            exit(ret)

        print('--> Init runtime environment')
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')
        
    def run(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs)
    
        return result

def initRKNN(model_path, core_id=RKNNLite.NPU_CORE_AUTO):
    rknn_lite = RKNNLite()
    
    # Load RKNN Model
    ret = rknn_lite.load_rknn(model_path)
    if ret != 0:
        print("Load RKNN model failed")
        exit(ret)
    
    # Initialize runtime environment
    ret = rknn_lite.init_runtime(core_mask=core_id)
    if ret != 0:
        print("Init runtime environment failed")
        exit(ret)
    
    print(f'{model_path} initialized successfully')
    return rknn_lite

def initRKNNs(model_path, num_instances=3):
    rknn_list = []
    for i in range(num_instances):
        core_id = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2][i % 3]
        rknn_list.append(initRKNN(model_path, core_id))
    return rknn_list

class RKNNPoolExecutor:
    def __init__(self, model_path, num_instances, func):
        self.num_instances = num_instances
        self.queue = Queue()
        self.rknn_pool = initRKNNs(model_path, num_instances)
        self.pool = ThreadPoolExecutor(max_workers=num_instances)
        self.func = func
        self.num = 0

    def put(self, frame):
        self.queue.put(self.pool.submit(self.func, self.rknn_pool[self.num % self.num_instances], frame))
        self.num += 1

    def get(self):
        if self.queue.empty():
            return None, False
        fut = self.queue.get()
        return fut.result(), True

    def release(self):
        self.pool.shutdown()
        for rknn_lite in self.rknn_pool:
            rknn_lite.release()

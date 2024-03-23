class Logs:
    def __init__(self):
        self.reset()
        self.iter_callback = None
        self.eval_callback = None

    def reset(self):
        self.loss = None
        self.lrs = None
        self.data_time = None
        self.iter_time = None
        self.cuda_memory = 0

        self.evaluation_metrics = None

        self.iter_idx = 0
        self.epoch = 0
        
    def log_evaluation(self, stats):
        if len(stats) != 12:
            raise ValueError("Expected 12 COCO stats, got {}".format(len(stats)))
        # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.018
        # Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.025
        # Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.018
        # Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
        # Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.002
        # Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.027
        # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.080
        # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.153
        # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.547
        # Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
        # Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.200
        # Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.545
        key_map = {
            0: "AP@IoU=0.50:0.95|maxDets=100",
            1: "AP@IoU=0.50|maxDets=100",
            2: "AP@IoU=0.75|maxDets=100",
            6: "AR@IoU=0.50:0.95|maxDets=1",
            7: "AR@IoU=0.50:0.95|maxDets=10",
            8: "AR@IoU=0.50:0.95|maxDets=100",
        }
        metrics = {}
        for i, key in key_map.items():
            metrics[key] = stats[i]
        self.evaluation_metrics = metrics
        
        if self.eval_callback is not None:
            self.eval_callback(self)

    def log_train_iter(self):
        if self.iter_callback is not None:
            self.iter_callback(self)


# Singleton object
LOGS = Logs()
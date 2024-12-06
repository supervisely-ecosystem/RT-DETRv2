import os
from dotenv import load_dotenv
import supervisely as sly
from supervisely.nn.benchmark import ObjectDetectionBenchmark, InstanceSegmentationBenchmark
import os

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

gt_project_id = 41774
# gt_dataset_ids = [64270]
# model_session = "http://localhost:8000"
model_session = 69289

# 1. Initialize benchmark
bench = ObjectDetectionBenchmark(api, gt_project_id, classes_whitelist=["person", "car", "cup"])
bench.api.retry_count = 1

# 2. Run evaluation
# This will run inference with the model and calculate metrics.
# Evaluation results will be saved in the "./benchmark" directory locally.
bench.run_evaluation(model_session=model_session)

# gt_path = "benchmark/COCO-100 (det) - rtdetrv2_r18vd_120e_coco_rerun_48.1.pth/gt_project"
# dt_path = "benchmark/COCO-100 (det) - rtdetrv2_r18vd_120e_coco_rerun_48.1.pth/pred_project"
# bench._evaluate(gt_path, dt_path, test=True)
key_metrics = bench.key_metrics
print(key_metrics)

# 3. Generate charts and dashboards
# This will generate visualization files and save them locally.
bench.visualize()

# 4. Upload to Supervisely Team Files
# To open the generated visualizations in the web interface, you need to upload them to Team Files.
bench.upload_visualizations(f"/model-benchmark/test")

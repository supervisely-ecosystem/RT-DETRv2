import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import yaml
from PIL import Image
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from boxmot import ByteTrack, BotSort
import supervisely as sly
import matplotlib.colors as mcolors
from dotenv import load_dotenv

from video import video_to_frames_ffmpeg, frames_to_video_ffmpeg

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))


def load_tracker(
        device: str,
        tracker_settings: dict = None,
        half: bool = False,
        per_class: bool = False
    ):
    if "cuda" in device and ":" not in device:
        device = "cuda:0"

    tracker_config = TRACKER_CONFIGS / 'botsort.yaml'

    # Load configuration from file
    with open(tracker_config, "r") as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        tracker_args = {param: details['default'] for param, details in yaml_config.items()}
    tracker_args.update(tracker_settings or {})
    tracker_args['per_class'] = per_class

    reid_weights = 'osnet_x1_0_msmt17.pt'
    if device == 'cpu':
        reid_weights = 'osnet_x0_5_msmt17.pt'

    reid_args = {
        'reid_weights': Path(reid_weights),
        'device': device,
        'half': half,
    }

    tracker_args.update(reid_args)
    tracker = BotSort(**tracker_args)
    if hasattr(tracker, 'model'):
        tracker.model.warmup()
    return tracker


def ann_to_detections(ann: sly.Annotation, cls2label: dict):
    # convert ann to N x (x, y, x, y, conf, cls) np.array
    detections = []
    for label in ann.labels:
        cat = cls2label[label.obj_class.name]
        bbox = label.geometry.to_bbox()
        conf = label.tags.get("confidence").value
        detections.append([bbox.left, bbox.top, bbox.right, bbox.bottom, conf, cat])
    detections = np.array(detections)
    return detections


def create_video_annotation(
        frame_to_annotation: dict,
        tracking_results: list,
        frame_shape: tuple,
        cat2obj: dict,      
    ):
    img_h, img_w = frame_shape
    video_objects = {}  # track_id -> VideoObject
    frames = []
    for (i, ann), tracks in zip(frame_to_annotation.items(), tracking_results):
        frame_figures = []
        for track in tracks:
            # crop bbox to image size
            dims = np.array([img_w, img_h, img_w, img_h]) - 1
            track[:4] = np.clip(track[:4], 0, dims)
            x1, y1, x2, y2, track_id, conf, cat = track[:7]
            cat = int(cat)
            track_id = int(track_id)
            rect = sly.Rectangle(y1, x1, y2, x2)
            video_object = video_objects.get(track_id)
            if video_object is None:
                obj_cls = cat2obj[cat]
                video_object = sly.VideoObject(obj_cls)
                video_objects[track_id] = video_object
            frame_figures.append(sly.VideoFigure(video_object, rect, i))
        frames.append(sly.Frame(i, frame_figures))

    objects = list(video_objects.values())
    video_ann = sly.VideoAnnotation(
        img_size=frame_shape,
        frames_count=len(frame_to_annotation),
        objects=sly.VideoObjectCollection(objects),
        frames=sly.FrameCollection(frames),
    )
    return video_ann


def generate_bright_color(hue=None):
    hue = np.random.random() if hue is None else hue
    saturation = np.random.uniform(0.8, 1.0)
    value = np.random.uniform(0.8, 1.0)
    rgb = mcolors.hsv_to_rgb([hue, saturation, value])
    return tuple(int(x * 255) for x in rgb)


if __name__ == '__main__':
    work_dir = 'output'
    device = 'cuda:0'
    video_path = f"./video.mp4"
    model_server = "http://localhost:8000"
    tracker_type = 'botsort'
    tracker_settings = {
        "track_high_thresh": 0.6,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.7,
        "match_thresh": 0.8,
    }

    os.makedirs(work_dir, exist_ok=True)
    api = sly.Api()
    session = sly.nn.inference.Session(api, session_url=model_server)
    model_meta = session.get_model_meta()
    tracker = load_tracker(device, tracker_settings)
    name2cat = {x.name: i for i, x in enumerate(model_meta.obj_classes)}
    cat2obj = {i: obj for i, obj in enumerate(model_meta.obj_classes)}

    # Break video into frames
    frames_dir = f"{work_dir}/frames"
    video_to_frames_ffmpeg(video_path, frames_dir)

    # Track
    track_results = []
    # predictions = {}
    img_paths = sorted(Path(frames_dir).glob("*.jpg"), key=lambda x: x.name)[:10]
    for i, img_path in enumerate(tqdm(img_paths)):
        ann = session.inference_image_path(img_path)
        detections = ann_to_detections(ann, name2cat)  # N x (x, y, x, y, conf, cls)
        img = Image.open(img_path).convert("RGB")
        img = np.asarray(img)
        tracks = tracker.update(detections, img)  # M x (x, y, x, y, track_id, conf, cls, det_id)
        track_results.append(tracks)
        # predictions[i] = ann

    # Draw tracks on image
    os.makedirs(f"{work_dir}/tracks", exist_ok=True)
    track_count = max([int(track[4]) for tracks in track_results for track in tracks]) + 1
    colors = [generate_bright_color(i / track_count) for i in range(track_count)]
    np.random.shuffle(colors)
    for i, img_path in enumerate(img_paths):
        img = np.array(Image.open(img_path).convert("RGB"))
        tracks = track_results[i]
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cat = track[:7]
            x1, y1, x2, y2 = map(round, [x1, y1, x2, y2])
            cat = int(cat)
            class_name = cat2obj[cat].name
            track_id = int(track_id)
            color = colors[track_id]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{class_name} {track_id}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        img = Image.fromarray(img)
        img.save(f"{work_dir}/tracks/{i + 1:06d}.jpg")

    # Create Video from frames
    frames_to_video_ffmpeg(
        input_pattern=f"{work_dir}/tracks/%06d.jpg",
        output_path=f"{work_dir}/output.mp4",
        fps=10,
        crf=23,
        preset="medium"
    )
    
    # Create VideoAnnotation
    # frame_shape = img.size[::-1]
    # video_ann = create_video_annotation(predictions, track_results, frame_shape, cat2obj)

    # Upload to the platform
    # ...
import os
from typing import List, Optional

import supervisely as sly
from pycocotools.coco import COCO
from supervisely.app.widgets import Button, Card, Container, ReloadableArea, TrainValSplits

import supervisely_integration.train.globals as g
import supervisely_integration.train.ui.parameters as parameters

trainval_container = Container()
trainval_area = ReloadableArea(trainval_container)

select_button = Button("Select splits")
change_button = Button("Change splits")
change_button.hide()

card = Card(
    title="4️⃣ Select splits",
    description="Select splits for training and validation",
    collapsable=True,
    content=Container([trainval_area, select_button]),
    content_top_right=change_button,
    lock_message="Click on the Change splits button to select other splits",
)
card.lock()
card.collapse()


def init_splits(project_fs: Optional[int] = None):
    if not project_fs:
        trainval_container._widgets.clear()
    else:
        trainval_splits = TrainValSplits(project_fs=project_fs)
        trainval_container._widgets.append(trainval_splits)
    trainval_area.reload()


@select_button.click
def splits_selected():
    g.splits = trainval_container._widgets[0].get_splits()

    # sly.logger.info(f"Selected splits: {g.splits}")

    card.lock()
    card.collapse()
    change_button.show()

    parameters.card.unlock()
    parameters.card.uncollapse()

    create_trainval()


@change_button.click
def change_splits():
    g.splits = None
    g.converted_project = None

    g.train_dataset_path = None
    g.val_dataset_path = None

    sly.logger.debug("Splits reset.")

    card.unlock()
    card.collapse()
    change_button.hide()

    parameters.card.lock()
    parameters.card.collapse()


def create_trainval():
    train_items, val_items = g.splits
    train_items: List[sly.project.project.ItemInfo]
    val_items: List[sly.project.project.ItemInfo]

    converted_project_dir = os.path.join(g.CONVERTED_DIR, g.selected_project_info.name)
    sly.fs.mkdir(converted_project_dir)
    train_dataset_path = os.path.join(converted_project_dir, "train")
    val_dataset_path = os.path.join(converted_project_dir, "val")

    g.train_dataset_path = train_dataset_path
    g.val_dataset_path = val_dataset_path

    project_meta_path = os.path.join(converted_project_dir, "meta.json")
    sly.json.dump_json_file(g.project.meta.to_json(), project_meta_path)

    for items, dataset_path in zip(
        [train_items, val_items], [train_dataset_path, val_dataset_path]
    ):
        prepare_dataset(dataset_path, items)

    print(converted_project_dir)

    g.converted_project = sly.Project(converted_project_dir, sly.OpenMode.READ)
    sly.logger.info(f"Project created in {converted_project_dir}")

    for dataset_fs in g.converted_project.datasets:
        dataset_fs: sly.Dataset
        selected_classes = [obj_class.name for obj_class in g.selected_classes]

        coco_anno = get_coco_annotations(dataset_fs, g.converted_project.meta, selected_classes)
        coco_anno_path = os.path.join(dataset_fs.directory, "coco_anno.json")
        sly.json.dump_json_file(coco_anno, coco_anno_path)

    sly.logger.info("COCO annotations created")


def prepare_dataset(dataset_path: str, items: List[sly.project.project.ItemInfo]):
    img_dir = os.path.join(dataset_path, "img")
    ann_dir = os.path.join(dataset_path, "ann")
    sly.fs.mkdir(img_dir)
    sly.fs.mkdir(ann_dir)
    for item in items:
        src_img_path = item.img_path
        src_ann_path = item.ann_path
        dst_img_path = os.path.join(img_dir, item.name)
        dst_ann_path = os.path.join(ann_dir, f"{item.name}.json")
        sly.fs.copy_file(src_img_path, dst_img_path)
        sly.fs.copy_file(src_ann_path, dst_ann_path)

    sly.logger.info(f"Dataset prepared in {dataset_path}")


def get_coco_annotations(dataset: sly.Dataset, meta: sly.ProjectMeta, selected_classes: List[str]):
    coco_anno = {"images": [], "categories": [], "annotations": []}
    cat2id = {name: i for i, name in enumerate(selected_classes)}
    img_id = 1
    ann_id = 1
    for name in dataset.get_items_names():
        ann = dataset.get_ann(name, meta)
        img_dict = {
            "id": img_id,
            "height": ann.img_size[0],
            "width": ann.img_size[1],
            "file_name": name,
        }
        coco_anno["images"].append(img_dict)

        for label in ann.labels:
            if isinstance(label.geometry, (sly.Bitmap, sly.Polygon)):
                rect = label.geometry.to_bbox()
            elif isinstance(label.geometry, sly.Rectangle):
                rect = label.geometry
            else:
                continue
            class_name = label.obj_class.name
            if class_name not in selected_classes:
                continue
            x, y, x2, y2 = rect.left, rect.top, rect.right, rect.bottom
            ann_dict = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat2id[class_name],
                "bbox": [x, y, x2 - x, y2 - y],
                "area": (x2 - x) * (y2 - y),
                "iscrowd": 0,
            }
            coco_anno["annotations"].append(ann_dict)
            ann_id += 1

        img_id += 1

    coco_anno["categories"] = [{"id": i, "name": name} for name, i in cat2id.items()]
    # Test:
    coco_api = COCO()
    coco_api.dataset = coco_anno
    coco_api.createIndex()
    return coco_anno

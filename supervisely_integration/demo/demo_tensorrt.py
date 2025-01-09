import json
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
from rtdetrv2_pytorch.references.deploy.rtdetrv2_tensorrt import TRTInference


assert torch.cuda.is_available(), "TensorRT only supports GPU mode"
device = 'cuda'


engine_path = "model/best.engine"  # put your tensorrt model (*.engine) file here
model_meta_path = "model/model_meta.json"  # put your model meta file here
image_path = "img/coco_sample.jpg"  # put your image file here


def draw(images, labels, boxes, scores, classes, thrh = 0.5):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        for l, b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=str(classes[lab[l].item()]), fill='blue', )


if __name__ == "__main__":

    # load class names
    with open(model_meta_path, "r") as f:
        model_meta = json.load(f)
    classes = [c["title"] for c in model_meta["classes"]]

    # load model
    model = TRTInference(engine_path, device=device)
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    # prepare image
    im_pil = Image.open(image_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(device)
    im_data = transforms(im_pil)[None]

    # inference
    output = model({
        'images': im_data.to(device), 
        'orig_target_sizes': orig_size.to(device),
    })

    # save result
    draw([im_pil], output['labels'], output['boxes'], output['scores'], classes)
    im_pil.save("result.jpg")

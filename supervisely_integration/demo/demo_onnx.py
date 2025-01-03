import json
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
import onnxruntime as ort 

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
print("Using device:", ort.get_device())


onnx_path = "model/best.onnx"  # put your onnx model file here
model_meta_path = "model/model_meta.json"  # put your model meta file here
image_path = "img/coco_sample.jpg"  # put your image file here


def draw(images, labels, boxes, scores, classes, thrh = 0.5):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        for b in box:
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=str(classes[lab[i].item()]), fill='blue', )


if __name__ == "__main__":

    # load class names
    with open(model_meta_path, "r") as f:
        model_meta = json.load(f)
    classes = [c["title"] for c in model_meta["classes"]]

    # load model
    sess = ort.InferenceSession(onnx_path, providers=providers)
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    # prepare image
    im_pil = Image.open(image_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None]
    im_data = transforms(im_pil)[None]

    # inference
    output = sess.run(
        output_names=None,
        input_feed={'images': im_data.data.numpy(), "orig_target_sizes": orig_size.data.numpy()}
    )
    labels, boxes, scores = output

    # save result
    draw([im_pil], labels, boxes, scores, classes)
    im_pil.save("result.jpg")

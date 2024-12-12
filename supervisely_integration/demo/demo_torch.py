from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
from rtdetrv2_pytorch.src.core import YAMLConfig


device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = "model/best.pth"
config_path = "model/model_config.yml"
model_meta_path = "model/model_meta.json"
image_path = "img/coco_sample.jpg"


def draw(images, labels, boxes, scores, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]
        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(),2)}", fill='blue', )


if __name__ == "__main__":

    # load checkpoint
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    model = cfg.model
    model.load_state_dict(state)
    model.deploy().to(device)
    postprocessor = cfg.postprocessor.deploy().to(device)
    classes = ...
    h, w = 640, 640
    transforms = T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
    ])

    # prepare image
    im_pil = Image.open(image_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(device)
    im_data = transforms(im_pil)[None].to(device)

    # inference
    output = model(im_data, orig_size)
    labels, boxes, scores = output

    # save result
    draw([im_pil], labels, boxes, scores)
    im_pil.save("result.jpg")
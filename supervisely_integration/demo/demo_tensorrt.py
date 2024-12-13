from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
from rtdetrv2_pytorch.references.deploy.rtdetrv2_tensorrt import TRTInference


assert torch.cuda.is_available(), "TensorRT only supports GPU mode"
device = 'cuda'


engine_path = "model/best.engine"
image_path = "img/coco_sample.jpg"


def draw(images, labels, boxes, scores, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        for l, b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=str(lab[l].item()), fill='blue', )


if __name__ == "__main__":

    model = TRTInference(engine_path, device=device)

    im_pil = Image.open(image_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None]

    output = model({
        'images': im_data.to(device), 
        'orig_target_sizes': orig_size.to(device),
    })

    draw([im_pil], output['labels'], output['boxes'], output['scores'])
    im_pil.save("result.jpg")

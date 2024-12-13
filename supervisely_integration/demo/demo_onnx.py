from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
import onnxruntime as ort 

print("Using device:", ort.get_device())


onnx_path = "model/best.onnx"
image_path = "img/coco_sample.jpg"


def draw(images, labels, boxes, scores, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        for b in box:
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=str(lab[i].item()), fill='blue', )


if __name__ == "__main__":

    sess = ort.InferenceSession(onnx_path)

    im_pil = Image.open(image_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None]

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None]

    output = sess.run(
        output_names=None,
        input_feed={'images': im_data.data.numpy(), "orig_target_sizes": orig_size.data.numpy()}
    )

    labels, boxes, scores = output

    draw([im_pil], labels, boxes, scores)
    im_pil.save("result.jpg")

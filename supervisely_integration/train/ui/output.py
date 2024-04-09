from supervisely.app.widgets import Card, Container, FolderThumbnail, LineChart, Progress

import rtdetr_pytorch.train as train_cli

# TODO: Fix import, now it's causing error
# from rtdetr_pytorch.src.misc.sly_logger import Logs

train_loss = LineChart("Train Loss", series=[{"name": "Train loss", "data": []}])
train_loss_counter = 0

train_progress = Progress()

output_folder = FolderThumbnail()
output_folder.hide()


def iter_callback(logs):
    global train_loss_counter
    data = (train_loss_counter, logs.loss)
    train_loss.add_to_series("Train loss", data)
    train_loss_counter += 1


train_cli.setup_callbacks(iter_callback=iter_callback)

card = Card(
    title="Download the weights",
    description="Here you can download the weights of the trained model",
    content=Container([train_progress, train_loss, output_folder]),
)
card.lock()

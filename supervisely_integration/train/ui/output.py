from supervisely.app.widgets import Card, Container, FolderThumbnail, Progress

train_progress = Progress()

output_folder = FolderThumbnail()
output_folder.hide()

card = Card(
    title="6️⃣ Download the weights",
    description="Here you can download the weights of the trained model",
    collapsable=True,
    content=Container([train_progress, output_folder]),
)
card.lock()

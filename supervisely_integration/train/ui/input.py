import supervisely as sly
from supervisely.app.widgets import Card, Container, ProjectThumbnail  # SelectProject, Button

import supervisely_integration.train.globals as g
import supervisely_integration.train.ui.model as model

# select_project = SelectProject(g.project_id, g.wotkspace_id, compact=True)
# select_project_button = Button("Select project")
# change_project_button = Button("Change project")
# change_project_button.hide()

project_thumbnail = ProjectThumbnail()
project_thumbnail.hide()

card = Card(
    title="1️⃣ Selected project",
    description="The project that will be used for training.",
    # collapsable=True,
    content=Container(
        [
            # select_project,
            # select_project_button,
            project_thumbnail,
        ]
    ),
    # content_top_right=change_project_button,
    # lock_message="Click on the Change project button to select another project",
)


# @select_project_button.click
def project_selected():
    # g.selected_project_id = select_project.get_selected_id()
    g.selected_project_id = g.project_id
    g.selected_project_info = g.api.project.get_info_by_id(g.selected_project_id)
    sly.logger.info(
        f"Selected project: {g.selected_project_info.name} with ID: {g.selected_project_info.id}"
    )
    g.selected_project_meta = sly.ProjectMeta.from_json(
        g.api.project.get_meta(g.selected_project_id)
    )

    sly.logger.info("Project meta saved into globals.")

    project_thumbnail.set(g.selected_project_info)
    project_thumbnail.show()
    # change_project_button.show()
    # card.lock()

    model.card.unlock()


project_selected()

# @change_project_button.click
# def change_project():
#     g.selected_project_id = None
#     g.selected_project_info = None
#     g.project_dir = None
#     g.project = None

#     sly.logger.info("Project selection reset.")

#     project_thumbnail.set(None)
#     project_thumbnail.hide()
#     change_project_button.hide()
#     card.unlock()

#     model.card.lock()

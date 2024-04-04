import supervisely as sly
from supervisely.app.widgets import Card, ProjectThumbnail

import supervisely_integration.train.globals as g

g.project_info = g.api.project.get_info_by_id(g.project_id)
g.project_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(g.project_id))
sly.logger.info("Project meta saved into globals.")
sly.logger.info(f"Selected project: {g.project_info.name} with ID: {g.project_info.id}")
project_thumbnail = ProjectThumbnail(g.project_info)

card = Card(
    title="Selected project",
    description="The project that will be used for training.",
    content=project_thumbnail,
)

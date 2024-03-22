from supervisely.app.widgets import Button, Card, Container, Editor, RadioTabs

import supervisely_integration.train.ui.output as output

general_tab = Container()
checkpoints_tab = Container()
optimization_tab = Container()

parameters_editor = Editor(language_mode="yaml", height_lines=100)
advanced_tab = Container([parameters_editor])

run_button = Button("Run training")
stop_button = Button("Stop training", button_type="danger")
stop_button.hide()


parameters_tabs = RadioTabs(
    ["General", "Checkpoints", "Optimization", "Advanced"],
    contents=[
        general_tab,
        checkpoints_tab,
        optimization_tab,
        Container([general_tab, checkpoints_tab, optimization_tab, advanced_tab]),
    ],
)

card = Card(
    title="5️⃣ Training hyperparameters",
    description="Specify training hyperparameters using one of the methods.",
    collapsable=True,
    content=Container(
        [parameters_tabs, run_button],
    ),
    content_top_right=stop_button,
)
card.lock()
card.collapse()


@run_button.click
def run_training():
    output.card.unlock()
    output.card.uncollapse()

    # TODO: Implement the training process


@stop_button.click
def stop_training():
    # TODO: Implement the stop process
    pass

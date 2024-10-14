from typing import List

from supervisely.app.widgets import Container, Empty, Field, Input, InputNumber, Switch

from supervisely_integration.train.ui.utils import (
    OrderedWidgetWrapper,
    create_linked_getter,
    get_switch_value,
    set_switch_value,
)

schedulers = [("empty", "Without scheduler")]

by_epoch_input = Switch(True)
by_epoch_field = Field(by_epoch_input, "By epoch")

# Step scheduler
step_scheduler = OrderedWidgetWrapper("StepLR")
step_scheduler.add_input(
    "by_epoch",
    by_epoch_input,
    by_epoch_field,
    get_switch_value,
    set_switch_value,
)

step_input = InputNumber(3, 1, step=1)
step_field = Field(step_input, "LR sheduler step")
step_scheduler.add_input(
    "step_size",
    step_input,
    wraped_widget=step_field,
)

step_gamma_input = InputNumber(0.1, 0, step=1e-5, size="small")
step_gamma_field = Field(step_gamma_input, "Gamma")
step_scheduler.add_input("gamma", step_gamma_input, step_gamma_field)
schedulers.append((repr(step_scheduler), "Step LR"))


# Multistep
def get_multisteps(input_w: Input) -> List[int]:
    steps: str = input_w.get_value()
    return [int(st.strip()) for st in steps.split(",")]


def set_multisteps(input_w: Input, value: List[int]):
    input_w.set_value(",".join(value))


multi_steps_scheduler = OrderedWidgetWrapper("MultiStepLR")
multi_steps_scheduler.add_input(
    "by_epoch",
    by_epoch_input,
    by_epoch_field,
    get_switch_value,
    set_switch_value,
)

multi_steps_input = Input("16,22")
multi_steps_field = Field(
    multi_steps_input,
    "LR sheduler steps",
    "Many int step values splitted by comma",
)
multi_steps_scheduler.add_input(
    "milestones",
    multi_steps_input,
    wraped_widget=multi_steps_field,
    custom_value_getter=get_multisteps,
    custom_value_setter=set_multisteps,
)

multi_steps_gamma_input = InputNumber(0.1, 0, step=1e-5, size="small")
multi_steps_gamma_field = Field(multi_steps_gamma_input, "Gamma")
multi_steps_scheduler.add_input("gamma", multi_steps_gamma_input, multi_steps_gamma_field)
schedulers.append((repr(multi_steps_scheduler), "Multistep LR"))

# exponential
exp_scheduler = OrderedWidgetWrapper("ExponentialLR")
exp_scheduler.add_input(
    "by_epoch",
    by_epoch_input,
    by_epoch_field,
    get_switch_value,
    set_switch_value,
)

exp_gamma_input = InputNumber(0.1, 0, step=1e-5, size="small")
exp_gamma_field = Field(exp_gamma_input, "Gamma")
exp_scheduler.add_input("gamma", exp_gamma_input, exp_gamma_field)
schedulers.append((repr(exp_scheduler), "Exponential LR"))


# reduce on plateau
reduce_plateau_scheduler = OrderedWidgetWrapper("ReduceOnPlateauLR")
reduce_plateau_scheduler.add_input(
    "by_epoch",
    by_epoch_input,
    by_epoch_field,
    get_switch_value,
    set_switch_value,
)

reduce_plateau_factor_input = InputNumber(0.1, 0, step=1e-5, size="small")
reduce_plateau_factor_field = Field(
    reduce_plateau_factor_input,
    "Factor",
    "Factor by which the learning rate will be reduced. new_param = param * factor",
)
reduce_plateau_scheduler.add_input(
    "factor",
    reduce_plateau_factor_input,
    reduce_plateau_factor_field,
)

reduce_plateau_patience_input = InputNumber(10, 2, step=1, size="small")
reduce_plateau_patience_field = Field(
    reduce_plateau_patience_input,
    "Patience",
    "Number of epochs with no improvement after which learning rate will be reduced",
)
reduce_plateau_scheduler.add_input(
    "patience",
    reduce_plateau_patience_input,
    reduce_plateau_patience_field,
)

schedulers.append((repr(reduce_plateau_scheduler), "ReduceOnPlateau LR"))

# CosineAnnealingLR
cosineannealing_scheduler = OrderedWidgetWrapper("CosineAnnealingLR")
cosineannealing_scheduler.add_input(
    "by_epoch",
    by_epoch_input,
    by_epoch_field,
    get_switch_value,
    set_switch_value,
)

cosineannealing_tmax_input = InputNumber(1, 1, step=1, size="small")
cosineannealing_tmax_field = Field(
    cosineannealing_tmax_input,
    "T max",
    "Maximum number of iterations",
)
cosineannealing_scheduler.add_input(
    "T_max",
    cosineannealing_tmax_input,
    cosineannealing_tmax_field,
)

etamin_switch_input = Switch(True)
etamin_input = InputNumber(0, 0, step=1e-6, size="small")
etamin_field = Field(
    Container([etamin_switch_input, etamin_input]),
    "Min LR",
    "Minimum learning rate",
)

etamin_ratio_input = InputNumber(0, 0, step=1e-6, size="small")
etamin_ratio_input.disable()
etamin_ratio_field = Field(
    etamin_ratio_input,
    "Min LR Ratio",
    "The ratio of the minimum parameter value to the base parameter value",
)

cosineannealing_scheduler.add_input(
    "eta_min",
    etamin_input,
    etamin_field,
    custom_value_getter=create_linked_getter(
        etamin_input,
        etamin_ratio_input,
        etamin_switch_input,
        True,
    ),
)

cosineannealing_scheduler.add_input(
    "eta_min_ratio",
    etamin_ratio_input,
    etamin_ratio_field,
    custom_value_getter=create_linked_getter(
        etamin_input,
        etamin_ratio_input,
        etamin_switch_input,
        False,
    ),
)
schedulers.append((repr(cosineannealing_scheduler), "Cosine Annealing LR"))


# CosineRestartLR
cosinerestart_scheduler = OrderedWidgetWrapper("CosineRestartLR")
cosinerestart_scheduler.add_input(
    "by_epoch",
    by_epoch_input,
    by_epoch_field,
    get_switch_value,
    set_switch_value,
)

cosinerestart_preiods_input = Input("1")
cosinerestart_preiods_field = Field(
    cosinerestart_preiods_input,
    "Periods",
    "Periods for each cosine anneling cycle. Many int step values splitted by comma",
)
cosinerestart_scheduler.add_input(
    "periods",
    cosinerestart_preiods_input,
    wraped_widget=cosinerestart_preiods_field,
    custom_value_getter=get_multisteps,
    custom_value_setter=set_multisteps,
)

cosinerestart_restart_weights_input = Input("1")
cosinerestart_restart_weights_field = Field(
    cosinerestart_restart_weights_input,
    "Restart weights",
    "Periods for each cosine anneling cycle. Many int step values splitted by comma",
)

cosinerestart_scheduler.add_input(
    "restart_weights",
    cosinerestart_restart_weights_input,
    wraped_widget=cosinerestart_restart_weights_field,
    custom_value_getter=get_multisteps,
    custom_value_setter=set_multisteps,
)

cosinerestart_scheduler.add_input(
    "eta_min",
    etamin_input,
    etamin_field,
    custom_value_getter=create_linked_getter(
        etamin_input,
        etamin_ratio_input,
        etamin_switch_input,
        True,
    ),
)

cosinerestart_scheduler.add_input(
    "eta_min_ratio",
    etamin_ratio_input,
    etamin_ratio_field,
    custom_value_getter=create_linked_getter(
        etamin_input,
        etamin_ratio_input,
        etamin_switch_input,
        False,
    ),
)
schedulers.append((repr(cosinerestart_scheduler), "Cosine Restart LR"))

# LinearLR
linear_scheduler = OrderedWidgetWrapper("LinearLR")
linear_scheduler.add_input(
    "by_epoch",
    by_epoch_input,
    by_epoch_field,
    get_switch_value,
    set_switch_value,
)

linear_start_factor_input = InputNumber(0.333, 1e-4, step=1e-4)
linear_start_factor_field = Field(
    linear_start_factor_input,
    "Start factor",
    description=(
        "The number we multiply learning rate in the "
        "first epoch. The multiplication factor changes towards end_factor "
        "in the following epochs"
    ),
)
linear_scheduler.add_input("start_factor", linear_start_factor_input, linear_start_factor_field)


linear_end_factor_input = InputNumber(1.0, 1e-4, step=1e-4)
linear_end_factor_field = Field(
    linear_end_factor_input,
    "End factor",
    description=("The number we multiply learning rate at the end " "of linear changing process"),
)
linear_scheduler.add_input("end_factor", linear_end_factor_input, linear_end_factor_field)
schedulers.append((repr(linear_scheduler), "Linear LR"))

# PolyLR

poly_scheduler = OrderedWidgetWrapper("PolyLR")
poly_scheduler.add_input(
    "by_epoch",
    by_epoch_input,
    by_epoch_field,
    get_switch_value,
    set_switch_value,
)

poly_eta_input = InputNumber(0, 0, step=1e-6, size="small")
poly_eta_field = Field(poly_eta_input, "Min LR", "Minimum learning rate at the end of scheduling")
poly_scheduler.add_input(
    "eta_min",
    poly_eta_input,
    poly_eta_field,
)

poly_power_input = InputNumber(1, 1e-3, step=1e-1, size="small")
poly_power_field = Field(poly_power_input, "Power", "The power of the polynomial")
poly_scheduler.add_input(
    "power",
    poly_power_input,
    poly_power_field,
)
schedulers.append((repr(poly_scheduler), "Polynomial LR"))

schedulers_params = {
    "Without scheduler": Empty(),
    repr(step_scheduler): step_scheduler,
    repr(multi_steps_scheduler): multi_steps_scheduler,
    repr(exp_scheduler): exp_scheduler,
    repr(cosineannealing_scheduler): cosineannealing_scheduler,
    repr(reduce_plateau_scheduler): reduce_plateau_scheduler,
    repr(cosinerestart_scheduler): cosinerestart_scheduler,
    repr(linear_scheduler): linear_scheduler,
    repr(poly_scheduler): poly_scheduler,
    # repr(onecycle_scheduler): onecycle_scheduler,
}

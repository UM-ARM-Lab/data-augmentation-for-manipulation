from state_space_dynamics import unconstrained_dynamics_nn


def get_model(model_class_name):
    if model_class_name == "SimpleNN":
        return unconstrained_dynamics_nn.UnconstrainedDynamicsNN

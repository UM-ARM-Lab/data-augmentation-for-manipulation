from pytorch_lightning import Callback


class HeartbeatCallback(Callback):
    def __init__(self, scenario):
        self.scenario = scenario

    def on_train_batch_start(self, *args, **kwargs):
        self.scenario.heartbeat()

    def on_validation_batch_start(self, *args, **kwargs):
        self.scenario.heartbeat()
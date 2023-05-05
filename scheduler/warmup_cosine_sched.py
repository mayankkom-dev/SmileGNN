import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule, CosineDecay


class WarmupCosineDecay(LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        warmup_steps: int,
        name: str = None,
    ):
        super(WarmupCosineDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.cosine_decay = CosineDecay(initial_learning_rate, decay_steps - warmup_steps)
        self.name = name

    def __call__(self, step):
        linear_factor = tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
        warmup_learning_rate = self.initial_learning_rate * linear_factor
        cosine_learning_rate = self.cosine_decay(step - self.warmup_steps)
        return tf.cond(step < self.warmup_steps, lambda: warmup_learning_rate, lambda: cosine_learning_rate)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
            "name": self.name,
        }

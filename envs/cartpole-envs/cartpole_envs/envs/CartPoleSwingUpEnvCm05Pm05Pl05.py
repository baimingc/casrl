from .cartpole_swingup_temp import *


class CartPoleSwingUpEnvCm05Pm05Pl05(CartPoleSwingUpEnv_template):
    def __init__(self):
        super().__init__( masscart =0.5, masspole=0.5, polelength=0.5)
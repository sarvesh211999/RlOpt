import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.misc import SlimFC


class PolicyNetwork(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,
                                       model_config, name)

        self.layer_1 = SlimFC(
            in_size=768, out_size=360, activation_fn="linear")
        self.layer_2 = SlimFC(
            in_size=360, out_size=180, activation_fn="relu")
        self.layer_3 = SlimFC(
            in_size=180, out_size=90, activation_fn="relu")
        self.layer_4 = SlimFC(
            in_size=90, out_size=self.num_outputs, activation_fn="linear")
        self.values = SlimFC(in_size=90, out_size=1, activation_fn="linear")

        self._last_value = None

    
    def forward(self, input_dict, state, seq_lens):
        features = input_dict["obs"]
        out = self.layer_1(features)
        out = self.layer_2(out)
        out = self.layer_3(out)
        final_out = self.layer_4(out)
        self._last_value = self.values(out)
        return final_out , []

    def value_function(self):
        return torch.squeeze(self._last_value, -1)
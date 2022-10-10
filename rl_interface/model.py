import math
import torch
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

from tiramisu_programs.surrogate_model_utils.modeling import Model_Recursive_LSTM_Embedding

train_device_name = 'cpu'  # choose training/storing device, either 'cuda:X' or 'cpu'
store_device_name = 'cpu'

store_device = torch.device(store_device_name)
train_device = torch.device(train_device_name)

BIG_NUMBER = 1e10

torch, nn = try_import_torch()


class TiramisuModelMult(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **kwargs):

        # print("in model init")

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name, **kwargs)

        nn.Module.__init__(self)

        shared_layer_sizes = model_config["custom_model_config"]["layer_sizes"]
        input_size = model_config["custom_model_config"]["input_size"]
        embedding_size = shared_layer_sizes[-1]
        prev_layer_size = embedding_size * 2 + 26

        # Embedding
        self.prog_embedding = Model_Recursive_LSTM_Embedding(input_size=input_size,
                                                            comp_embed_layer_sizes=shared_layer_sizes)


        #Outputs
        #1 Policy
        self._logits = SlimFC(
            in_size=prev_layer_size,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )

        #2 Value
        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_len):
        # print("in forward")
        # print('OBS:',input_dict['obs_flat'])
        obs = input_dict["obs_flat"]["representation"][-1, :, :]
        #OBS needs to be flattened because it has a shape of (5,1052) initially, as specified in the observation space
        obs = torch.flatten(obs)

        #computation embedding layer
        comps_embeddings = self._comp_embd_layers(obs)
        ## print("from comp embd layer: ",comps_embeddings.size(),comps_embeddings)

        #recursive loop embedding layer
        loops_tensor = input_dict["obs_flat"]["loops_representation"]
        child_list = input_dict["obs_flat"]["child_list"][:][:][0][0]
        x  = [child_list, obs, loops_tensor]

        # prediction layer
        self._features = self.prog_embedding(x)
        ## print("from prediction embd layer: ",self._features.size())
        logits = self._logits(self._features)
        logits = logits - BIG_NUMBER * (1 -
                                        input_dict["obs_flat"]["action_mask"])

        # print("\nLa distribution de probabilté est: ", F.softmax(logits))

        self._value = self._value_branch(self._features)

        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        # print(self._value)
        return self._value.squeeze(1)

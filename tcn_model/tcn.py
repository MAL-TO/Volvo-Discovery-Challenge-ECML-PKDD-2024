import torch
import torch.nn as nn
import pytorch_tcn


class MyTCN(nn.Module):
    def __init__(self, num_input_channels, num_classes=3, tcn_config=None):
        super(MyTCN, self).__init__()
        
        self.num_input_channels = num_input_channels
        self.num_classes = num_classes

        if tcn_config == None:
            tcn_config = [
                #   input_channels,             kernel_size,    dialation,  num_channels,   output_size 
                [   self.num_input_channels,    2,              1,          128,            32          ],
                [   32,                         2,              2,          128,            32          ],
                [   32,                         2,              4,          128,            32          ],
                [   32,                         2,              8,          128,            32          ]
            ]

        self.tcn = self.__build_tcn(tcn_config)
        
        self.mlp = nn.Sequential(
            nn.Linear( sum([x[-1] for x in tcn_config]), 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),

            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.25),

            nn.Linear(32, num_classes)
        )

    def __build_tcn(self, config):
        tcn_layers = []
        for i, layer in enumerate(config):
            input_channels, kernel_size, dialation, num_channels, output_size = layer 
            tcn_layer =  pytorch_tcn.TCN(
                                num_inputs = input_channels,
                                num_channels = [num_channels],
                                kernel_size= kernel_size,
                                dilations = [dialation],
                                output_projection = output_size,

                                use_norm = 'batch_norm',
                                use_skip_connections = True,
                                causal = True,
                                input_shape='NLC' # batch, timesteps, features
                        )
            tcn_layers.append(tcn_layer)

        return tcn_layers

    def __forward_tcn(self, x):
        outputs = []
        print("TCN INPUT SHAPE:", x.shape)

        out = x.copy()
        for layer in self.tcn:
            out = layer(out)
            outputs.append(out)
        
        result = torch.concat(outputs, dim=-1)
        print("TCN OUTPUT SHAPE:", result.shape)
        return result
    
    def forward(self, timeseries, static_info=None):
        # temporal_features_tensor = (BATCH, TIMESTAMP, FEATURES)
        temporal_features_tensor = self.__forward_tcn(timeseries)

        # Version 1, more better assai, va testato
        init_shape = temporal_features_tensor.shape
        reshaped_temporal_features = temporal_features_tensor.reshape(-1, init_shape[-1]) # (BATCH x TIMESTAMP, TEMP_FEATURES)
        reshaped_classes = self.mlp(reshaped_temporal_features) # (BATCH x TIMESTAMP, CLASS_PROBS)
        output = reshaped_classes.reshape((*init_shape[:-1], self.num_classes)) # (BATCH, TIMESTAMP, CLASS_PROBS)

        # # Version 2, slower, but so che funziona
        # results = []
        # for batch_el in temporal_features_tensor:
        #     res = self.mlp(batch_el)
        #     results.append(res)
        # output = torch.stack(results,dim=0) # (BATCH, TIMESTAMP, CLASS_PROBS)

        return output

    def print_devices(self):
        print(list(self.image_feature_extractor.parameters())[0].device)
        print(list(self.tcn.parameters())[0].device)
        print(list(self.regressor.parameters())[0].device)


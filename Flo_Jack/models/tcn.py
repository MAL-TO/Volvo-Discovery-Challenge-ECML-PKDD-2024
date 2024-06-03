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
                [  2048,                         3,              1,         4096,           -1        ],
                [  4096,                         3,              2,         4096,           -1        ],
                [  4096,                         3,              3,         4096,           -1        ],
                [  4096,                         3,              4,         4096,           -1        ],
                # [  1024,                         3,              5,         2048,           1024        ],
                # [  1024,                         4,              8,         2048,           1024        ],
                # [  1024,                         2,             16,         2048,           1024        ]
            ]

        self.tcn = self.__build_tcn(tcn_config)
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.num_input_channels, 4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.2),
            
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.2),
            
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.BatchNorm1d(512),
            # nn.Dropout(0.1),
        )


        self.mlp = nn.Sequential(
            nn.Linear( 12 + sum([x[-2] if x[-1] == -1 else x[-1] for x in tcn_config]), 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.25),
            

            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),

            nn.Linear(128, num_classes)
        )

    def __build_tcn(self, config):
        tcn_layers = nn.ModuleList([])
        for i, layer in enumerate(config):
            input_channels, kernel_size, dialation, num_channels, output_size = layer 
            tcn_layer =  pytorch_tcn.TCN(
                                num_inputs = input_channels,
                                num_channels = [num_channels],
                                kernel_size= kernel_size,
                                dilations = [dialation],
                                output_projection = output_size if output_size != -1 else None,

                                use_norm = 'batch_norm',
                                use_skip_connections = True,
                                causal = False,
                                input_shape='NLC' # batch, timesteps, features
                        )
            tcn_layers.append(tcn_layer)

        return tcn_layers

    def __forward_tcn(self, x):
        outputs = []
        #print("TCN INPUT SHAPE:", x.shape)

        out = x
        for layer in self.tcn:
            out = layer(out)
            #TODO: check if a copy() is needed
            outputs.append(out)
        
        result = torch.concat(outputs, dim=-1)
        #print("TCN OUTPUT SHAPE:", result.shape)
        return result
    
    def forward(self, timeseries, static_info=None):
        init_shape = timeseries.shape
        reshaped_timeseries = timeseries.reshape(-1, init_shape[-1]) # (BATCH x TIMESTAMP, INPUT)
        reshaped_features = self.feature_extractor(reshaped_timeseries) # (BATCH x TIMESTAMP, FEATURES)
        ts_features = reshaped_features.reshape((*init_shape[:-1], 2048)) # (BATCH, TIMESTAMP, FEATURES)


        # temporal_features_tensor = (BATCH, TIMESTAMP, FEATURES)
        temporal_features_tensor = self.__forward_tcn(ts_features)
        temporal_features_and_variants_tensor = torch.cat([temporal_features_tensor, static_info], dim=-1)

        # Version 1, more better assai, va testato
        init_shape = temporal_features_and_variants_tensor.shape
        reshaped_temporal_features = temporal_features_and_variants_tensor.reshape(-1, init_shape[-1]) # (BATCH x TIMESTAMP, TEMP_FEATURES)
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


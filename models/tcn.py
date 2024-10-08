import torch
import torch.nn as nn
import pytorch_tcn
from torch.nn import functional as F
from torch.jit import Final

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class AttentionHead(nn.Module):
    def __init__(self, in_size, out_size, bias=True, dropout=0.25):
        super(AttentionHead, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        # self.ls = LayerScale(size)
        self.prenorm = nn.LayerNorm(in_size)
        self.q = nn.Linear(in_size, out_size, bias=bias)
        self.k = nn.Linear(in_size, out_size, bias=bias)
        self.v = nn.Linear(in_size, out_size, bias=bias)
        self.norm1 = nn.LayerNorm(out_size)
        self.dropout1 = nn.Dropout(dropout)
        self.input_projection = nn.Linear(in_size, out_size) if in_size != out_size else nn.Identity()

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(out_size, out_size),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )
        self.norm2 = nn.LayerNorm(out_size)
        
    def forward(self, x):
        # x = (BATCH, TIMESTEP, FEATURES)
        x = self.prenorm(x)

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        score = torch.bmm(q, k.transpose(1, 2)) / (self.out_size**0.5)
        attention = F.softmax(score, dim=-1)
        weighted = torch.bmm(attention, v)

        output = self.dropout1(weighted + self.input_projection(x))
        output = self.norm1(output)
        output = self.linear_net(output) + output
        output = self.norm2(output)

        return output


class SS_TCN(nn.Module):
    def __init__(self, num_input_channels, is_phase_1, apply_feature_extractor=True, num_classes=3, hidden_features=64, tcn_config=None):
        super(SS_TCN, self).__init__()
        
        self.num_input_channels = num_input_channels
        self.num_classes = num_classes
        self.apply_feature_extractor = apply_feature_extractor
        self.is_phase_1 = is_phase_1

        if tcn_config == None:
            tcn_config = [
                #   input_channels,                                                 kernel_size,    dialation,  num_channels,   output_size 
                [  512 if self.apply_feature_extractor else num_input_channels,    3,              1,         hidden_features,           -1        ],
                [   hidden_features,                                               3,              2,         hidden_features,           -1        ],
                [   hidden_features,                                               3,              3,         hidden_features,           -1        ],
                [   hidden_features,                                               3,              4,         hidden_features,           -1        ]
            ]

        self.tcn = self.__build_tcn(tcn_config)
        
        self.prenorm = nn.LayerNorm(self.num_input_channels)

        if self.apply_feature_extractor:
            self.feature_extractor = nn.Sequential(
                nn.Linear(self.num_input_channels, 4096),
                nn.ReLU(),
                nn.BatchNorm1d(4096),
                nn.Dropout(0.2),
                
                nn.Linear(4096, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2)
            )

        self.tcn_output_dim = hidden_features
        self.attention_layer = nn.Linear(self.tcn_output_dim, 1)

        
        self.attention_dim = 512 if self.apply_feature_extractor else num_input_channels
        self.attention_heads = AttentionHead(in_size=self.attention_dim, out_size=self.tcn_output_dim) 
        # self.attention_heads = nn.MultiheadAttention(embed_dim=self.attention_dim, vdim=self.tcn_output_dim, num_heads=1, batch_first=True)
        
        self.cross_attention = nn.MultiheadAttention(self.tcn_output_dim, num_heads=8, dropout=0.2,  bias=True, batch_first=True)
        self.cross_attention2 = nn.MultiheadAttention(self.tcn_output_dim, num_heads=8, dropout=0.2, bias=True, batch_first=True)

        self.input_dim_mlp = 2*self.tcn_output_dim
        self.mlp = nn.Sequential(
            nn.Linear(12+self.input_dim_mlp, 4*self.input_dim_mlp),
            nn.ReLU(),
            nn.BatchNorm1d(4*self.input_dim_mlp),
            nn.Dropout(0.25),
            

            nn.Linear(4*self.input_dim_mlp, self.input_dim_mlp),
            nn.ReLU(),
            nn.BatchNorm1d(self.input_dim_mlp),
            nn.Dropout(0.25),


            nn.Linear(self.input_dim_mlp, min(int(self.input_dim_mlp//4), 512)),
            nn.ReLU(),
            nn.BatchNorm1d(min(int(self.input_dim_mlp//4), 512)),
            nn.Dropout(0.25),

            nn.Linear(min(int(self.input_dim_mlp//4), 512), min(int(self.input_dim_mlp//8), 128)),
            nn.ReLU(),
            nn.BatchNorm1d(min(int(self.input_dim_mlp//8), 128)),
            nn.Dropout(0.25),

            nn.Linear(min(int(self.input_dim_mlp//8), 128), num_classes)
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

                                use_norm = 'layer_norm',
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
        for i in range(len(self.tcn)):
            layer = self.tcn[i]
            out = layer(out)
            outputs.append(out)
        
        # result = torch.concat(outputs, dim=-1)
        # result = outputs[-1]
        result = torch.stack(outputs, dim=0).sum(dim=0)

        #print("TCN OUTPUT SHAPE:", result.shape)
        return result
    
    def forward(self, timeseries, static_info=None):
        timeseries = self.prenorm(timeseries)

        if static_info is None:
            static_info = torch.zeros((*timeseries.shape[:-1],12)).to('cuda' if torch.cuda.is_available() else 'cpu')

        if self.apply_feature_extractor:
            init_shape = timeseries.shape
            reshaped_timeseries = timeseries.reshape(-1, init_shape[-1]) # (BATCH x TIMESTAMP, INPUT)
            reshaped_features = self.feature_extractor(reshaped_timeseries)  # (BATCH x TIMESTAMP, FEATURES)
            ts_features = reshaped_features.reshape((*init_shape[:-1], 512)) # (BATCH, TIMESTAMP, FEATURES)
        else:
            ts_features = timeseries

        # temporal_features_tensor = (BATCH, TIMESTAMP, FEATURES)
        temporal_features_tensor = self.__forward_tcn(ts_features)
        temporal_features_tensor = temporal_features_tensor *  torch.softmax( self.attention_layer(temporal_features_tensor), dim = -1)

        # attention
        attention_features_tensor = self.attention_heads(ts_features)

        # Cross attention between TCN and Transformer
        temporal_features_crossattention, _ = self.cross_attention(temporal_features_tensor, attention_features_tensor, attention_features_tensor)
        attention_features_crossattention, _  =  self.cross_attention2(attention_features_tensor, temporal_features_tensor, temporal_features_tensor)
        temporal_features_tensor = temporal_features_tensor + temporal_features_crossattention
        attention_features_tensor = attention_features_tensor + attention_features_crossattention

        # concat attention and tf
        temporal_features_tensor = torch.cat([temporal_features_tensor, attention_features_tensor], dim=-1)

        # temporal_features_and_variants_tensor = (BATCH, TIMESTAMP, TEMP_FEATURES)
        #print(temporal_features_tensor.shape, static_info.shape)
        temporal_features_tensor = temporal_features_tensor.mean(dim=1)
        temporal_features_and_variants_tensor = torch.cat([temporal_features_tensor, static_info], dim=-1)

        # fully connected with healthy/unlhealthy output 
        # AVERAGE POOLING GLOBAL TO BE SURE THAT THE OUTPUT IS INDEPENDENT FROM THE TIME SERIES LENGTH
        # (BATCH, TEMP_FEATURES)

        # global pooling over each time series 
        # (BATCH, TEMP_FEATURES)
        if self.is_phase_1:
            output = self.mlp(temporal_features_and_variants_tensor)
            output = torch.nn.functional.softmax(output, dim=-1)
        else: 
            output = self.mlp(temporal_features_and_variants_tensor)
            output = torch.sigmoid(output)

        return output

    def print_devices(self):
        print(list(self.image_feature_extractor.parameters())[0].device)
        print(list(self.tcn.parameters())[0].device)
        print(list(self.regressor.parameters())[0].device)
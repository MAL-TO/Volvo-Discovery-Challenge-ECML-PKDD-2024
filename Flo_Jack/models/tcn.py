import torch
import torch.nn as nn
import pytorch_tcn
from torch.nn import functional as F
from torch.jit import Final


class Attention(nn.Module):
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    fast_attn: Final[bool]

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fast_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")  # FIXME

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        #3, B, n_head, N, head_dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        #B, n_head, N, head_dim
        q, k, v = qkv.unbind(0)
        #B, n_head, N, head_dim
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fast_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class MyTCN(nn.Module):
    def __init__(self, num_input_channels, num_classes=3, tcn_config=None):
        super(MyTCN, self).__init__()
        
        self.num_input_channels = num_input_channels
        self.num_classes = num_classes

        if tcn_config == None:
            tcn_config = [
                #   input_channels,             kernel_size,    dialation,  num_channels,   output_size 
                [  2048,                         3,              1,         2048,           -1        ],
                [  2048,                         3,              2,         2048,           -1        ],
                [  2048,                         3,              3,         2048,           -1        ],
                [  2048,                         3,              4,         2048,           -1        ],
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

        self.attention_block = Block(dim = sum([x[-2] if x[-1] == -1 else x[-1] for x in tcn_config]), num_heads=1)
        
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
        #attention 
        temporal_features_tensor = self.attention_block(temporal_features_tensor)
        # temporal_features_and_variants_tensor = (BATCH, TIMESTAMP, TEMP_FEATURES)
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


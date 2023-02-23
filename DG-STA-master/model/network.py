from .st_att_layer import *
import torch.nn as nn
import torch

class DG_STA(nn.Module):
    def __init__(self, num_classes, dp_rate):
        super(DG_STA, self).__init__()

        h_dim = 32
        h_num= 8

        self.input_map = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            LayerNorm(128),
            nn.Dropout(dp_rate),
        )
        self.NN1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            LayerNorm(128),
            nn.Dropout(dp_rate),
        )
        self.NN2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            LayerNorm(256),

        )
        #input_size, h_num, h_dim, dp_rate, time_len, domain
        self.s_att = ST_ATT_Layer(input_size=128,output_size= 128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, domain="spatial", time_len = 8)


        self.t_att = ST_ATT_Layer(input_size=128, output_size= 128,h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, domain="temporal", time_len = 8)

        self.cls = nn.Linear(512, num_classes)


    # def forward(self, x):
    #     # input shape: [batch_size, time_len, joint_num, 3]
    #
    #     time_len = x.shape[1]
    #     joint_num = x.shape[2]
    #
    #     #reshape x
    #     x = x.reshape(-1, time_len * joint_num,3)
    #
    #     #input map
    #     x = self.input_map(x)
    #
    #     #temporal
    #     x = self.t_att(x)
    #
    #     #spatal
    #     x = self.s_att(x)
    #
    #
    #
    #
    #
    #     x = x.sum(1) / x.shape[1]
    #     pred = self.cls(x)
    #     return pred
    #
    def forward(self, x):
        # input shape: [batch_size, time_len, joint_num, 3]

        time_len = x.shape[1]
        joint_num = x.shape[2]

        #reshape x
        x = x.reshape(-1, time_len * joint_num,3)

        #input map
        x = self.input_map(x)

        #temporal
        temp_spatial_x = self.t_att(x)

        temp_spatial_x = self.NN1(temp_spatial_x)
        #spatal
        temp_spatial_x = self.s_att(temp_spatial_x)

        temp_spatial_x = self.NN1(temp_spatial_x)
        # temp_spatial_x = temp_spatial_x.sum(1) / temp_spatial_x.shape[1]


        #spatal
        spatial_temp_x = self.s_att(x)

        spatial_temp_x = self.NN1(spatial_temp_x)
        #temporal
        spatial_temp_x = self.t_att(spatial_temp_x)

        spatial_temp_x = self.NN1(spatial_temp_x)


        # spatial_temp_x = spatial_temp_x.sum(1) / spatial_temp_x.shape[1]

        muting_x = torch.concat((temp_spatial_x,spatial_temp_x),2)

        MLP_x = self.NN2(x)

        muting_x = torch.concat((muting_x,MLP_x),2)

        x = muting_x.sum(1) / muting_x.shape[1]

        pred = self.cls(x)


        return pred
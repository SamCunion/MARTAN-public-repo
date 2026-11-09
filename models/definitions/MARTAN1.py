#MARTAN-1 architecture. Takes a probable tornado sample, centred, and reduced to 50x50 matrix. Takes two-tilt DBZ, VEL, RHOHV and WIDTH data, as well as mesoanalysis variables to produce an intensity estimation.
import torch
import torch.nn as NN

INPUT_CHANNELS = 2 #per input type, corresponds to the number of tilts in MARTAN-1
INPUT_TYPES = 4 #DBZ,VEL,RHOHV,WIDTH

#defines the radar input segments of the NN
def radar_input_head():
    return NN.Sequential(
        NN.Conv2d(INPUT_CHANNELS, 16, kernel_size=3, padding=1),
        NN.Conv2d(16, 32, kernel_size=3, padding=1),
        NN.BatchNorm2d(32),
        NN.ReLU(True),
        NN.MaxPool2d(2),

        NN.Conv2d(32, 64, kernel_size=3, padding=1),
        NN.BatchNorm2d(64),
        NN.ReLU(True),
        NN.MaxPool2d(2)
    )

#shared learning, merges the three input heads
def shared_segment():
    return NN.Sequential(
        NN.Conv2d(64 * INPUT_TYPES, 128, kernel_size=3, padding=1),
        NN.BatchNorm2d(128),
        NN.ReLU(True),
        NN.MaxPool2d(2),

        NN.Conv2d(128, 256, kernel_size=3, padding=1),
        NN.BatchNorm2d(256),
        NN.ReLU(True),
        NN.MaxPool2d(2)
    )

#shared fully connected layer(s)
def fc_segment():
    return NN.Sequential(
        #256 from radar
        NN.Linear(256, 64),
        NN.ReLU(True),
        NN.Dropout(0.2),

        NN.Linear(64, 32),
        NN.ReLU(True),
    )

#tornado probability head
def prob_head():
    return NN.Sequential(
        NN.Linear(32, 1)
    )

class MARTAN_1(NN.Module):

    #specifies the variables that should be accessed for GRAD-CAM analysis
    gradcam_targets = ["DBZ_Head", "VEL_Head", "RHOHV_Head", "WIDTH_Head"]

    def __init__(self):
        super().__init__()

        #make input heads
        self.DBZ_Head = radar_input_head()
        self.VEL_Head = radar_input_head()
        self.RHOHV_Head = radar_input_head()
        self.WIDTH_Head = radar_input_head()

        #shared layer(s)
        self.shared = shared_segment()

        #linearise the data
        self.linearise = NN.Sequential(
            NN.AdaptiveAvgPool2d((1, 1)),
            NN.Flatten()
        )

        #FC layer
        self.fc = fc_segment()

        #tornado prob head
        self.tor_prob = prob_head()

    
    #forward pass through MARTAN-1
    def forward(self, DBZ, VEL, RHOHV, WIDTH):
        #each input is in the form (batch, 2, 120, 100)
        f1 = self.DBZ_Head(DBZ)
        f2 = self.VEL_Head(VEL)
        f3 = self.RHOHV_Head(RHOHV)
        f4 = self.WIDTH_Head(WIDTH)
        #each is now (batch, 32, 60, 30) #RECALCULATE ALL THESE

        #merge each head's channels
        combined_heads = torch.cat([f1, f2, f3, f4], dim=1)
        #shape: (batch, 96, 60, 30)
        shared_conv = self.shared(combined_heads)
        #shape: (batch, 64, 30, 15)
        linear_radar = self.linearise(shared_conv)
        #shape: (batch, 64)
        fc = self.fc(linear_radar)
        #shape: (batch, 32)

        #output heads
        out_prob = self.tor_prob(fc) #shape: (batch, 5)

        #output
        return out_prob
    

#CoordConv - Liu et al. https://arxiv.org/pdf/1807.03247
class CoordConv(NN.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = NN.Conv2d(in_channels + 2, out_channels, kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        b, c, h, w = x.shape

        #one-dimensional coordinate axis
        xs = torch.linspace(-1.0, 1.0, w, device=x.device)
        ys = torch.linspace(-1.0, 1.0, h, device=x.device)

        #convert to 2d coordinates
        x_coords = xs.unsqueeze(0).repeat(h, 1)
        y_coords = ys.unsqueeze(1).repeat(1, w)

        #construct the coordinates for every member of the batch
        x_coords = x_coords.unsqueeze(0).repeat(b, 1, 1, 1)
        y_coords = y_coords.unsqueeze(0).repeat(b, 1, 1, 1)

        #combine the coordinate channels with the radar inputs
        out = torch.cat([x, x_coords, y_coords], dim=1)
        return self.conv(out)
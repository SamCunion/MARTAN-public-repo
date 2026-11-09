#MARTAN-0 architecture. Takes lowest-tilt VEL, x, x data and creates a tornado probability prediction, and WSOL tornado segmentation

#MARTAN-0 v0.2.0

import torch.nn as NN
import torch

INPUT_CHANNELS = 2 #VEL

#defines the input segments of the NN
def conv_layers():
    return NN.Sequential(
        NN.Conv2d(INPUT_CHANNELS, 16, kernel_size=3, padding=0),
        NN.Conv2d(16, 32, kernel_size=3, padding=0),
        NN.BatchNorm2d(32),
        NN.ReLU(True),
        NN.MaxPool2d(2),

        NN.Conv2d(32, 64, kernel_size=3, padding=1),
        NN.BatchNorm2d(64),
        NN.ReLU(True),
        NN.MaxPool2d(2),

        NN.Conv2d(64, 128, kernel_size=3, padding=1),
        NN.BatchNorm2d(128),
        NN.ReLU(True),
        NN.MaxPool2d(2),

        NN.Conv2d(128, 256, kernel_size=3, padding=1),
        NN.BatchNorm2d(256),
        NN.ReLU(True),
        NN.MaxPool2d(2),
    )

#shared fully connected layer(s)
def fc_segment():
    return NN.Sequential(
        NN.Linear(256, 64),
        NN.ReLU(),
        NN.Dropout(0.2),
    )

#tornado probability head
def prob_head():
    return NN.Sequential(
        NN.Linear(64, 1),
    )

class MARTAN_0(NN.Module):

    def __init__(self):
        super().__init__()

        self.gradients = None
        self.activations = None
        self.with_grad = False

        #make conv layers
        self.input_head = conv_layers()

        #linearise the data
        self.linearise = NN.Sequential(
            NN.AdaptiveAvgPool2d((1, 1)),
            NN.Flatten()
        )

        #FC layer
        self.fc = fc_segment()

        #tornado probability head
        self.tor_prob = prob_head()
        
    #allow GRAD-CAM to be output
    def with_gradcam(self):
            self.with_grad = True
            FINAL_CONV = next(layer for layer in reversed(getattr(self, "input_head")) if isinstance(layer, torch.nn.Conv2d))
            def forward_hook(module, input, output):
                self.activations = output.detach()

            def backward_hook(module, input, output):
                self.gradients = output[0].detach()
            
            FINAL_CONV.register_forward_hook(forward_hook)
            FINAL_CONV.register_full_backward_hook(backward_hook)

    #forward pass through MARTAN-0
    def forward(self, input_stack):
        #shape: (BATCH, 1, 100, 120)


        conv_layer = self.input_head(input_stack)
        #shape: (BATCH, 256, 30, 60) #CALCULATE AGAIN

        linear = self.linearise(conv_layer)
        #shape: (BATCH, 256)

        fc = self.fc(linear)
        #shape: (BATCH, 64)

        #output heads
        out_prob = self.tor_prob(fc) #shape: (batch, 1)

        #gradcam output
        GRADCAM = None
        if (self.with_grad): #gradcam mode activated
            b, c, a, r = input_stack.shape
            self.zero_grad()
            out_prob.backward()

            A = self.activations
            dY_dA = self.gradients
            d2Y_dA2 = dY_dA ** 2
            d3Y_dA3 = dY_dA ** 3
            eps = 1e-8

            numerator = d2Y_dA2
            denominator = 2 * d2Y_dA2 + A * d3Y_dA3
            denominator = torch.where(denominator != 0, denominator, torch.tensor(eps))

            alpha = numerator / denominator

            weights = (alpha * torch.nn.functional.relu(dY_dA)).sum(dim=(2,3))

            cam = (weights[:, :, None, None] * A).sum(dim=1)
            cam = torch.nn.functional.relu(cam)
            cam -= cam.min()
            cam /= cam.max() + 1e-8

            cam = torch.nn.functional.interpolate(
                cam.unsqueeze(0),
                size=(a, r),
                mode="bilinear",
                align_corners=True
            )[0]

            return out_prob, cam

        #output
        return out_prob
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancementNetNoPool(nn.Module):

    def __init__(self):
        super(EnhancementNetNoPool, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        num_filters = 32

        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(num_filters * 2, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(num_filters * 2, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(num_filters * 2, 24, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))

        fused1 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        fused2 = self.relu(self.conv6(torch.cat([x2, fused1], dim=1)))

        residuals = torch.tanh(self.conv7(torch.cat([x1, fused2], dim=1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(residuals, 3, dim=1)

        enhance1 = x + r1 * (x ** 2 - x)
        enhance2 = enhance1 + r2 * (enhance1 ** 2 - enhance1)
        enhance3 = enhance2 + r3 * (enhance2 ** 2 - enhance2)
        enhanced_1 = enhance3 + r4 * (enhance3 ** 2 - enhance3)

        refine1 = enhanced_1 + r5 * (enhanced_1 ** 2 - enhanced_1)
        refine2 = refine1 + r6 * (refine1 ** 2 - refine1)
        refine3 = refine2 + r7 * (refine2 ** 2 - refine2)
        enhanced_final = refine3 + r8 * (refine3 ** 2 - refine3)

        return enhanced_1, enhanced_final, residuals

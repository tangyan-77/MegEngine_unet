import megengine.module as M
import megengine.functional as F


class DoubleConv(M.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = M.Sequential(
            M.Conv2d(in_ch, out_ch, 3, padding=1),
            M.BatchNorm2d(out_ch),
            M.ReLU(),
            M.Conv2d(out_ch, out_ch, 3, padding=1),
            M.BatchNorm2d(out_ch),
            M.ReLU())

    def forward(self, input):
        return self.conv(input)


class Unet(M.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 32)
        self.conv2 = DoubleConv(32, 64)
        self.conv3 = DoubleConv(64, 128)
        self.conv4 = DoubleConv(128, 256)
        self.conv5 = DoubleConv(256, 512)

        self.pool = M.MaxPool2d(2)
        self.dropout = M.Dropout(0.3)

        self.conv6 = DoubleConv(768, 256)
        self.conv7 = DoubleConv(384, 128)
        self.conv8 = DoubleConv(192, 64)
        self.conv9 = DoubleConv(96, 32)
        self.conv10 = M.Conv2d(32, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2 = self.conv2(p1)
        p2 = self.pool(c2)

        c3 = self.conv3(p2)
        p3 = self.pool(c3)

        c4 = self.conv4(p3)
        p4 = self.pool(c4)

        c5 = self.conv5(p4)
        # d5 = self.dropout(c5)  # 崩

        up_6 = F.interpolate(c5, [32, 32], align_corners=False)

        merge6 = F.concat([up_6, c4], axis=1)

        c6 = self.conv6(merge6)

        up_7 = F.interpolate(c6, [64, 64], align_corners=False)
        merge7 = F.concat([up_7, c3], axis=1)
        c7 = self.conv7(merge7)

        up_8 = F.interpolate(c7, [128, 128], align_corners=False)
        merge8 = F.concat([up_8, c2], axis=1)
        c8 = self.conv8(merge8)

        up_9 = F.interpolate(c8, [256, 256], align_corners=False)
        merge9 = F.concat([up_9, c1], axis=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)

        return c10

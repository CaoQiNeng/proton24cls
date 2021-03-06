from common  import *
# import pretrainedmodels

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, downsample=None,
                 dropout_rate = 0.5):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.stride = stride
        self.kernel_size = kernel_size
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=self.kernel_size, stride=self.stride,
                     padding=int(self.kernel_size / 2), bias=False)
        self.bn2 =  nn.BatchNorm1d(out_planes)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(out_planes, out_planes, kernel_size=self.kernel_size, stride=1,
                     padding=int(self.kernel_size / 2))
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Net(nn.Module):

    def __init__(self, in_planes, num_classes=1000, kernel_size=15, dropout_rate = 0.5):
        super(Net, self).__init__()
        self.dilation = 1
        self.in_planes = in_planes
        self.out_planes = 64
        self.stride = 1
        self.kernel_size = kernel_size

        # pre conv
        self.conv1 = nn.Conv1d(self.in_planes, self.out_planes, kernel_size=self.kernel_size, stride=1, padding=int(self.kernel_size/2),
                               bias=False)
        self.in_planes = self.out_planes
        self.bn1 =  nn.BatchNorm1d(self.out_planes)
        self.relu = nn.ReLU(inplace=True)

        # first block
        self.conv2 = nn.Conv1d(self.out_planes, self.out_planes, kernel_size=self.kernel_size, stride=2, padding=int(self.kernel_size/2),
                               bias=False)
        self.bn2 = nn.BatchNorm1d(self.out_planes)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv3 = nn.Conv1d(self.out_planes, self.out_planes, kernel_size=self.kernel_size, stride=1, padding=int(self.kernel_size/2),
                               bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=self.kernel_size, stride=2, padding=int(self.kernel_size/2))

        layers = []
        for i in range(1, 16):
            if i % 4 == 0 :
                self.out_planes = self.in_planes + 64

            if i % 4 == 0 :
                downsample = nn.Sequential(
                    nn.Conv1d(self.in_planes, self.out_planes, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm1d(self.out_planes))
                self.stride = 2
            elif i % 2 == 0 :
                downsample = self.maxpool
                self.stride = 2
            else :
                downsample = None
                self.stride = 1

            layers.append(BasicBlock(self.in_planes, self.out_planes, self.kernel_size, self.stride, downsample))

            self.in_planes = self.out_planes

        self.layers = nn.Sequential(*layers)

        self.bn3 = nn.BatchNorm1d(self.out_planes)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.out_planes, num_classes)

    def forward(self, x):
        # pre conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # first block
        identity = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = x + self.maxpool(identity)

        # res block x 15
        x = self.layers(x)

        x = self.bn3(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def run_check_net():
    net = Net(12, num_classes=9, dropout_rate = 0.5)

    input = torch.randn(1, 12, 72000)
    output = net(input)

    print(output.shape)

#https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
def metric(truth, predict):
    truth_for_cls = np.sum(truth, axis=0) + 1e-11
    predict_for_cls = np.sum(predict, axis=0) + 1e-11

    # TP
    count = truth + predict
    count[count != 2] = 0
    TP = np.sum(count, axis=0) / 2

    precision = TP / predict_for_cls
    recall = TP / truth_for_cls

    return precision, recall


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # run_check_basenet()
    run_check_net()


    print('\nsuccess!')



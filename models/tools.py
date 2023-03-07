from torch import nn

from models.ml_decoder import MLDecoder


class FastAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


def add_ml_decoder_head(model, num_classes=-1, num_of_groups=-1, decoder_embedding=768):
    if num_classes == -1:
        num_classes = model.num_classes
    if hasattr(model, "avgpool") and hasattr(model, "fc"):  # resnet
        num_features = model.fc.in_features
        model.avgpool = nn.Identity()
        del model.fc
        model.fc = MLDecoder(num_classes=num_classes, initial_num_features=num_features,
                             num_of_groups=num_of_groups, decoder_embedding=decoder_embedding)
    elif hasattr(model, "avgpool") and hasattr(model, "head"):  # swin transformer
        num_features = model.head.in_features
        model.avgpool = nn.Identity()
        del model.head
        model.head = MLDecoder(num_classes=num_classes, initial_num_features=num_features,
                               num_of_groups=num_of_groups, decoder_embedding=decoder_embedding)
    elif hasattr(model, "avgpool") and hasattr(model, "classifier"):  # mobilenet v2
        num_features = model.classifier.in_features
        model.avgpool = nn.Identity()
        del model.classifier
        model.classifier = MLDecoder(num_classes=num_classes, initial_num_features=num_features,
                                     num_of_groups=num_of_groups, decoder_embedding=decoder_embedding)
    elif hasattr(model, "gap") and hasattr(model, "linear"):  # repvgg
        num_features = model.linear.in_features
        model.gap = nn.Identity()
        del model.linear
        model.linear = MLDecoder(num_classes=num_classes, initial_num_features=num_features,
                                 num_of_groups=num_of_groups, decoder_embedding=decoder_embedding)
    else:
        print("model is not suited for ml-decoder")
        exit(-1)

    return model

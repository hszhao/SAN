import torch

from model.san import san
from util.complexity import get_model_complexity_info


with torch.cuda.device(0):
    model = san(sa_type=0, layers=[2, 1, 2, 4, 1], kernels=[3, 7, 7, 7, 7], num_classes=1000)
    # model = san(sa_type=0, layers=[3, 2, 3, 5, 2], kernels=[3, 7, 7, 7, 7], num_classes=1000)
    # model = san(sa_type=0, layers=[3, 3, 4, 6, 3], kernels=[3, 7, 7, 7, 7], num_classes=1000)
    # model = san(sa_type=1, layers=[2, 1, 2, 4, 1], kernels=[3, 7, 7, 7, 7], num_classes=1000)
    # model = san(sa_type=1, layers=[3, 2, 3, 5, 2], kernels=[3, 7, 7, 7, 7], num_classes=1000)
    # model = san(sa_type=1, layers=[3, 3, 4, 6, 3], kernels=[3, 7, 7, 7, 7], num_classes=1000)

    flops, params = get_model_complexity_info(model.cuda(), (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('Params/Flops: {}/{}'.format(params, flops))

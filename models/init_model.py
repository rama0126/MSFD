
from torchvision.models.video import r3d_18
from torchvision.models.video import r2plus1d_18
# from torchvision.models.video import swin3d_t, Swin3D_T_Weights
# from torchvision.models.video import swin3d_s, Swin3D_S_Weights
# from torchvision.models.video import swin3d_b, Swin3D_B_Weights
# from vit_pytorch.vivit import ViT as vivit

from torch import nn 
import torch

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')            
"""
### architecture_list
    - torchvision_r3d_18
    - torchvision_r2plus1d_18
    - torch_hub_slow_r50
    - torch_hub_slowfast_r50
    - torch_hub_i3d_r50
    - torch_hub_slowfast_r101
    - vit_torch_ViViT
    - FTCN_TT
    - FTCN_only
"""


def init_model_architecture__(architecture, pretrained, separate_classifier=False, separate_feature_extractor=False):
    model_arch = architecture
    pretrained = pretrained
    if separate_feature_extractor:
        return seperate_feature_extractor(model_arch,pretrained)
    elif separate_classifier:
        return seperate_classifier_(model_arch,pretrained)
    if model_arch == 'torchvision_r3d_18':
        if pretrained :
            model = r3d_18(pretrained=True)
        else :
            model = r3d_18(pretrained=False)
        model.fc = nn.Linear(512,1)
    elif model_arch == 'torchvision_r2plus1d_18':
        if pretrained :
            model = r2plus1d_18(pretrained=True)
        else :
            model = r2plus1d_18(pretrained=False)
        model.fc = nn.Linear(512,1)
    elif model_arch == 'torch_hub_slow_r50':
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=pretrained)
        model.blocks[-1].proj = nn.Linear(2048,1)
    elif model_arch == 'torch_hub_slowfast_r50':
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=pretrained)
        model.blocks[-1].proj = nn.Linear(2304,1)
    elif model_arch == 'torch_hub_i3d_r50' :
        model = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=pretrained)
        model.blocks[-1].proj = nn.Linear(2048,1)
    elif model_arch == 'torch_hub_slowfast_r101':
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r101', pretrained=pretrained)
        model.blocks[-1].proj = nn.Linear(2304,1)
    elif model_arch == 'vit_torch_ViViT':
        model = vivit(
            image_size = 224,
            frames = 32,
            image_patch_size = 16,
            frame_patch_size = 2,
            dim = 1024,
            num_classes = 1,
            spatial_depth = 6,
            temporal_depth = 6,
            heads = 8,
            mlp_dim = 2048,)
    elif model_arch == 'FTCN_TT' :
        import sys
        sys.path.append('./models/FTCN')
        from models.FTCN.ftcn import I3D8x8 as FTCN_TT
        model = FTCN_TT()
        
    elif model_arch == 'FTCN_only':
        import sys
        sys.path.append('./models/FTCN')
        from models.FTCN.ftcn import I3D8x8 as FTCN_TT
        model = FTCN_TT()
        stop_point = model.stop_point
        if stop_point == 5:
            model.resnet.head = nn.Sequential(
                nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Flatten(),
                nn.Linear(1024,1)
            )
        elif stop_point == 6:
            model.resnet.head = nn.Sequential(
                nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Flatten(),
                nn.Linear(2048,1)
            )
        else :
            print(stop_point)
            raise ValueError('Invalid stop_point value')
        import torch
        weights = torch.load('/workspace/CDVD/outputs/20251109_icarl_FTCN_only/0_FF/model_30.pth')
        backbone_weights = {}
        classifier_weights = {}
        for key, value in weights.items():
            if key.startswith('backbone.'):
                new_key = key[len('backbone.'):]
                backbone_weights[new_key] = value
            elif key.startswith('classifier.'):
                new_key = key[len('classifier.'):]
                classifier_weights[new_key] = value
        model.load_state_dict(backbone_weights)
    return model
def seperate_classifier_(model_arch,pretrained):
    if model_arch == 'torchvision_r3d_18':
        if pretrained :
            model = r3d_18(pretrained=True)
        else :
            model = r3d_18(pretrained=False)
        model.fc = nn.Identity()
        fc = nn.Linear(512,1)
    elif model_arch == 'torchvision_r2plus1d_18':
        if pretrained :
            model = r2plus1d_18(pretrained=True)
        else :
            model = r2plus1d_18(pretrained=False)
        model.fc = nn.Identity()
        fc  = nn.Linear(512,1)
    elif model_arch == 'torch_hub_slow_r50':
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=pretrained)
        model.blocks[-1].proj = nn.Identity()
        fc  = nn.Linear(2048,1)
    elif model_arch == 'torch_hub_slowfast_r50':
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=pretrained)
        model.blocks[-1].proj = nn.Identity()
        fc = nn.Linear(2304,1)
    elif model_arch == 'torch_hub_i3d_r50' :
        model = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=pretrained)
        model.blocks[-1].proj = nn.Identity()
        fc = nn.Linear(2048,1)

    elif model_arch == 'torch_hub_slowfast_r101':
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r101', pretrained=pretrained)
        fc = nn.Linear(2304,1)
        model.blocks[-1].proj = nn.Identity()
    elif model_arch == 'vit_torch_ViViT':
        model = vivit(
            image_size = 224,
            frames = 32,
            image_patch_size = 16,
            frame_patch_size = 2,
            dim = 1024,
            num_classes = 1,
            spatial_depth = 6,
            temporal_depth = 6,
            heads = 8,
            mlp_dim = 2048,)
        fc = model.head
        model.head = nn.Identity()
    elif model_arch == 'FTCN_TT' :
        import sys
        sys.path.append('./models/FTCN')
        from models.FTCN.ftcn import I3D8x8 as FTCN_TT
        model = FTCN_TT()
        fc = model.resnet.head
        model.resnet.head = nn.Identity()


    elif model_arch == 'FTCN_only':
        import sys
        sys.path.append('./models/FTCN')
        from models.FTCN.ftcn import I3D8x8 as FTCN_TT
        model = FTCN_TT()
        stop_point = model.stop_point
        if stop_point == 5:
            fc = nn.Sequential(
                nn.Linear(1024,1)
            )
        elif stop_point == 6:
            fc= nn.Sequential(
                nn.Linear(2048,1)
            )
        else :
            print(stop_point)
            raise ValueError('Invalid stop_point value')
        model.resnet.head = nn.Identity()
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv3d):
                print(f"{name}: kernel_size={module.kernel_size}, in_channels={module.in_channels}, out_channels={module.out_channels}")
        weights = torch.load('/workspace/CDVD/outputs/20251110_icarl_FTCN_only/0_FF/model_20.pth')
        backbone_weights = {}
        classifier_weights = {}
        for key, value in weights.items():
            if key.startswith('backbone.'):
                new_key = key[len('backbone.'):]
                backbone_weights[new_key] = value
            elif key.startswith('classifier.'):
                new_key = key[len('classifier.'):]
                classifier_weights[new_key] = value
        model.load_state_dict(backbone_weights)
        fc.load_state_dict(classifier_weights)
    return model, fc
def seperate_feature_extractor(model_arch,pretrained):
    """
    Returns the feature extractor and classifier separately, while feature extractor can extract spatial and temporal features.
    Args:
        model_arch (str): The architecture of the model.
        pretrained (bool): Whether to use a pretrained model.
    Returns:
        tuple: A tuple containing the feature extractor and classifier.
    Raises:
        ValueError: If the model architecture is not supported.
    """
    if model_arch == 'torchvision_r3d_18':
        if pretrained :
            model = r3d_18(pretrained=True)
        else :
            model = r3d_18(pretrained=False)
        modules = list(model.children())[:-2]
        feature_extractor = nn.Sequential(*modules)
        fc = nn.Linear(512,1)
    elif model_arch == 'torchvision_r2plus1d_18':
        if pretrained :
            model = r2plus1d_18(pretrained=True)
        else :
            model = r2plus1d_18(pretrained=False)
        modules = list(model.children())[:-2]
        feature_extractor = nn.Sequential(*modules)
        fc  = nn.Linear(512,1)
    elif model_arch == 'torch_hub_slow_r50':
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=pretrained)
        feature_extractor = nn.Sequential(*list(model.blocks[:-1]))
        fc  = nn.Linear(2048,1)
    elif model_arch == 'torch_hub_slowfast_r50':
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=pretrained)
        feature_extractor = nn.Sequential(*list(model.blocks[:-1]))
        fc = nn.Linear(2304,1)
    elif model_arch == 'torch_hub_i3d_r50' :
        model = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=pretrained)
        feature_extractor = nn.Sequential(*list(model.blocks[:-1]))
        fc = nn.Linear(2048,1)

    elif model_arch == 'torch_hub_slowfast_r101':
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r101', pretrained=pretrained)
        feature_extractor = nn.Sequential(*list(model.blocks[:-1]))
        fc = nn.Linear(2304,1)
    elif model_arch == 'vit_torch_ViViT':
        model = vivit(
            image_size = 224,
            frames = 32,
            image_patch_size = 16,
            frame_patch_size = 2,
            dim = 1024,
            num_classes = 1,
            spatial_depth = 6,
            temporal_depth = 6,
            heads = 8,
            mlp_dim = 2048,)
        feature_extractor = model.extract_features
        fc = nn.Linear(1024,1)
    elif model_arch == 'FTCN_TT' :
        import sys
        sys.path.append('./models/FTCN')
        from models.FTCN.ftcn import I3D8x8 as FTCN_TT
        model = FTCN_TT()
        feature_extractor = nn.Sequential(*list(model.resnet.children())[:-1])
        fc = nn.Linear(400,1)
    elif model_arch == 'FTCN_only':
        import sys
        sys.path.append('./models/FTCN')
        from models.FTCN.ftcn import I3D8x8 as FTCN_TT
        model = FTCN_TT()
        # Print 3D Convolution layer kernel shapes
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv3d):
                print(f"{name}: kernel_size={module.kernel_size}, in_channels={module.in_channels}, out_channels={module.out_channels}")
        
        stop_point = model.stop_point
        if stop_point == 5:
            feature_extractor = nn.Sequential(*list(model.resnet.children())[:-1] + [nn.AdaptiveAvgPool3d((1,1,1)), nn.Flatten()])
            fc = nn.Linear(1024,1)
        elif stop_point == 6:
            feature_extractor = nn.Sequential(*list(model.resnet.children())[:-1] + [nn.AdaptiveAvgPool3d((1,1,1)), nn.Flatten()])
            fc= nn.Linear(2048,1)
        else :
            print(stop_point)
            raise ValueError('Invalid stop_point value')
    return feature_extractor, fc

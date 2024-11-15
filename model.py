
import os,json
from urllib.parse import urlsplit

import torch
torch.set_printoptions(sci_mode=False)
# torch.multiprocessing.set_sharing_strategy('file_system')
torch.multiprocessing.set_sharing_strategy('file_descriptor')
import torch.nn as nn
import torchvision
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import set_layer_config
from timm.models import is_model, model_entrypoint, load_checkpoint
from timm.models.layers import to_2tuple
from transformers import CLIPModel, CLIPProcessor


BACKBONE_DICT = {
    'resnet50': 2048,
    'CLIP': 512,
    'PLIP': 512,
    'MobileNetV3': 1280,
    'mobilenetv3': 1280,
    'ProvGigaPath': 1536,
    # 'CONCH': 512,
    'CONCH': 768,
    'UNI': 1024,
    'CTransPath': 768
}


def load_cfg_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)

def load_model_config_from_hf(model_id: str):
    cached_file = './backbones/ProvGigaPath/config.json'

    hf_config = load_cfg_from_json(cached_file)
    if 'pretrained_cfg' not in hf_config:
        # old form, pull pretrain_cfg out of the base dict
        pretrained_cfg = hf_config
        hf_config = {}
        hf_config['architecture'] = pretrained_cfg.pop('architecture')
        hf_config['num_features'] = pretrained_cfg.pop('num_features', None)
        if 'labels' in pretrained_cfg:  # deprecated name for 'label_names'
            pretrained_cfg['label_names'] = pretrained_cfg.pop('labels')
        hf_config['pretrained_cfg'] = pretrained_cfg

    # NOTE currently discarding parent config as only arch name and pretrained_cfg used in timm right now
    pretrained_cfg = hf_config['pretrained_cfg']
    pretrained_cfg['hf_hub_id'] = model_id  # insert hf_hub id for pretrained weight load during model creation
    pretrained_cfg['source'] = 'hf-hub'

    # model should be created with base config num_classes if its exist
    if 'num_classes' in hf_config:
        pretrained_cfg['num_classes'] = hf_config['num_classes']

    # label meta-data in base config overrides saved pretrained_cfg on load
    if 'label_names' in hf_config:
        pretrained_cfg['label_names'] = hf_config.pop('label_names')
    if 'label_descriptions' in hf_config:
        pretrained_cfg['label_descriptions'] = hf_config.pop('label_descriptions')

    model_args = hf_config.get('model_args', {})
    model_name = hf_config['architecture']
    return pretrained_cfg, model_name, model_args


def split_model_name_tag(model_name: str, no_tag: str = ''):
    model_name, *tag_list = model_name.split('.', 1)
    tag = tag_list[0] if tag_list else no_tag
    return model_name, tag


def parse_model_name(model_name: str):
    if model_name.startswith('hf_hub'):
        # NOTE for backwards compat, deprecate hf_hub use
        model_name = model_name.replace('hf_hub', 'hf-hub')
    parsed = urlsplit(model_name)
    assert parsed.scheme in ('', 'timm', 'hf-hub')
    if parsed.scheme == 'hf-hub':
        # FIXME may use fragment as revision, currently `@` in URI path
        return parsed.scheme, parsed.path
    else:
        model_name = os.path.split(parsed.path)[-1]
        return 'timm', model_name


def create_model():
    model_name = 'hf_hub:prov-gigapath/prov-gigapath'
    model_source, model_name = parse_model_name(model_name)
    pretrained_cfg, model_name, model_args = load_model_config_from_hf(model_name)
    kwargs = {}
    if model_args:
        for k, v in model_args.items():
            kwargs.setdefault(k, v)
    create_fn = model_entrypoint(model_name)
    with set_layer_config(scriptable=None, exportable=None, no_jit=None):
        model = create_fn(
            pretrained=False,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=None,
            **kwargs
        )
    load_checkpoint(model, './backbones/ProvGigaPath/pytorch_model.bin')

    return model


class ConvStem(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True,
            output_fmt=None,
            bias=True,
            strict_img_size=True,
            dynamic_img_pad=False):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class STModel(nn.Module):
    def __init__(self, backbone='resnet50', dropout=0.25, num_outputs=24665):
        super().__init__()
        self.backbone = backbone
        if backbone == 'resnet50':
            self.backbone_model = torchvision.models.resnet50(pretrained=True)
            self.backbone_model.fc = nn.Identity()
            self.transform = None
            self.image_processor = None
        elif backbone == 'CONCH':
            # from conch.open_clip_custom import create_model_from_pretrained
            # self.backbone_model, self.image_processor = create_model_from_pretrained('conch_ViT-B-16','./backbones/CONCH_weights_pytorch_model.bin')
            # self.transform = None
            from timm.models.vision_transformer import VisionTransformer
            self.backbone_model = VisionTransformer(embed_dim=768, 
                                                    depth=12, 
                                                    num_heads=12, 
                                                    mlp_ratio=4,
                                                    img_size=448, 
                                                    patch_size=16,
                                                    num_classes=0,
                                                    dynamic_img_size=True)
            self.backbone_model.load_state_dict(torch.load('./backbones/CONCH_vision_weights_pytorch_model.bin', weights_only=True))
        elif backbone == 'UNI':
            self.backbone_model = timm.create_model(
                "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
            )
            self.backbone_model.load_state_dict(torch.load("./backbones/UNI_pytorch_model.bin", map_location="cpu", weights_only=True), strict=True)
            self.transform = create_transform(**resolve_data_config(self.backbone_model.pretrained_cfg, model=self.backbone_model))
        elif backbone == 'ProvGigaPath':
            self.backbone_model = create_model()  # timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        elif backbone == 'CLIP':
            self.backbone_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        elif backbone == 'PLIP':
            self.backbone_model = CLIPModel.from_pretrained("./backbones/vinid_plip")
            self.image_processor = CLIPProcessor.from_pretrained("./backbones/vinid_plip")
        elif backbone == 'CTransPath':
            from swin_transformer import SwinTransformer
            # self.backbone_model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
            model_kwargs = dict(
                patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                embed_layer=ConvStem, pretrained=False
            )
            self.backbone_model = SwinTransformer(**model_kwargs)
            self.backbone_model.head = nn.Identity()
            self.backbone_model.load_state_dict(torch.load("./backbones/ctranspath.pth", map_location="cpu", weights_only=True)['model'], strict=True)
        else:
            raise ValueError('error')

        self.rho = nn.Sequential(*[
            nn.Linear(BACKBONE_DICT[backbone], BACKBONE_DICT[backbone]), 
            nn.ReLU(), 
            nn.Dropout(dropout)
        ])

        self.fc = nn.Linear(BACKBONE_DICT[backbone], num_outputs)

        # self.initialize_weights()
        self.rho.apply(self._init_weights)
        self.fc.apply(self._init_weights)

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        
        if self.backbone in ['PLIP', 'CLIP']:
            h = self.backbone_model.get_image_features(x)
        elif self.backbone in ['resnet50', 'UNI', 'ProvGigaPath', 'CONCH', 'CTransPath']:
            h = self.backbone_model(x)

        h = self.rho(h)

        h = self.fc(h)

        # h = torch.tanh(h)

        return h

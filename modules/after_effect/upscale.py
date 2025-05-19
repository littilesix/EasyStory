import os
import torch
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
except:
    import tools.FakeModules
    from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils.download_util import load_file_from_url
from gfpgan import GFPGANer
from gfpgan.archs.gfpgan_bilinear_arch import GFPGANBilinear
from gfpgan.archs.gfpganv1_arch import GFPGANv1
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from .processor import AfterEffectProcessor,Frame


class GFPGANer_rewrite(GFPGANer):
    """Helper for restoration with GFPGAN.

    It will detect and crop faces, and then resize the faces to 512x512.
    GFPGAN is used to restored the resized faces.
    The background is upsampled with the bg_upsampler.
    Finally, the faces will be pasted back to the upsample background image.

    Args:
        model_path (str): The path to the GFPGAN model. It can be urls (will first download it automatically).
        upscale (float): The upscale of the final output. Default: 2.
        arch (str): The GFPGAN architecture. Option: clean | original. Default: clean.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        bg_upsampler (nn.Module): The upsampler for the background. Default: None.
    """

    def __init__(self, model_path, upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None, device=None):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler

        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        # initialize the GFP-GAN
        if arch == 'clean':
            self.gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'bilinear':
            self.gfpgan = GFPGANBilinear(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'original':
            self.gfpgan = GFPGANv1(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=True,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'RestoreFormer':
            from gfpgan.archs.restoreformer_arch import RestoreFormer
            self.gfpgan = RestoreFormer()
        # initialize face helper
        self.face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=self.device,
            model_rootpath=f'{os.getcwd()}/models/scale')

        if model_path.startswith('https://'):
            model_path = load_file_from_url(
                url=model_path, model_dir=f'{os.getcwd()}/models/scale', progress=True, file_name=None)
        loadnet = torch.load(model_path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        self.gfpgan.load_state_dict(loadnet[keyname], strict=True)
        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)


class VideoScaler:
    denoise_strength = 0.5
    outscale = 2
    model_name = "RealESRGAN_x4plus"

    def __init__(self,processor:AfterEffectProcessor,denoise_strength = None,outscale = None,model_name = None):

        processor.video_width = processor.video_width * self.outscale
        processor.video_height = processor.video_height * self.outscale

        device = "cuda"

        if torch.backends.mps.is_available():
            device = "mps"
        elif not torch.cuda.is_available():
            divice = "cpu"

        if outscale:self.outscale = outscale

        if denoise_strength: self.denoise_strength = denoise_strength

        if model_name:
            self.model_name = model_name

        self.fp32 = True

        self.tile = 0

        self.tile_pad = 10

        self.pre_pad = 0

        self.face_enhance = True

        self.model_name = self.model_name.split('.pth')[0]

        model,netscale,file_url,model_path = self.get_model_info(self.model_name)

        #print(model_path)

        if not os.path.isfile(model_path):
            for url in file_url:
                # model_path will be updated
                try:
                    model_path = load_file_from_url(url=url, model_dir=f'{os.getcwd()}/models/scale', progress=True, file_name=None)
                except:
                    model_path = load_file_from_url(url="https://github.moeyy.xyz/"+url, model_dir=f'{os.getcwd()}/models/scale', progress=True, file_name=None)
        dni_weight = None
        if self.model_name == 'realesr-general-x4v3' and self.denoise_strength != 1:
            wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            model_path = [model_path, wdn_model_path]
            dni_weight = [denoise_strength, 1 - denoise_strength]
        # restorer
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=not self.fp32,
            device=device,
        )

        if 'anime' in self.model_name and self.face_enhance:
            print('face_enhance is not supported in anime models, we turned this option off for you. '
              'if you insist on turning it on, please manually comment the relevant lines of code.')
            self.face_enhance = False
        if self.face_enhance:  # Use GFPGAN for face enhancement
            self.face_enhancer = GFPGANer_rewrite(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=self.outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=self.upsampler)  # TODO support custom device
        else:
            self.face_enhancer = None

    def rend(self,frame:Frame):
        if self.face_enhance:
            _, _, new_frame = self.face_enhancer.enhance(frame.image, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            new_frame, _ = self.upsampler.enhance(frame.image, outscale=self.outscale)
        return new_frame

    def get_model_info(self,model_name):
        if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
        elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
        elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
        elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
        elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
            ]
        return model,netscale,file_url,os.path.join('./models/scale', model_name + '.pth')



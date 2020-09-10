from dliplib.utils.models.skip import Skip
from dliplib.utils.models.unet import UNet
from dliplib.utils.models.iradonmap import IRadonMap


def get_unet_model(in_ch=1, out_ch=1, scales=5, skip=4, channels=(32, 32, 64, 64, 128, 128), use_sigmoid=True, use_norm=True):
    assert (1 <= scales <= 6)
    skip_channels = [skip] * (scales)
    return UNet(in_ch=in_ch, out_ch=out_ch, channels=channels[:scales],
                skip_channels=skip_channels, use_sigmoid=use_sigmoid, use_norm=use_norm)


def get_skip_model(in_ch=1, out_ch=1, channels=(128,) * 5, skip_channels=(4,) * 5):
    return Skip(in_ch=in_ch, out_ch=out_ch, channels=channels,
                skip_channels=skip_channels)


def get_iradonmap_model(ray_trafo, fully_learned, scales=5, skip=4,
                        use_sigmoid=True, coord_mat=None):
    post_process = get_unet_model(in_ch=1, out_ch=1, scales=scales, skip=skip,
                                  use_sigmoid=use_sigmoid)
    return IRadonMap(ray_trafo=ray_trafo, post_process=post_process,
                     fully_learned=fully_learned, coord_mat=coord_mat)

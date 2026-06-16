import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, Bottleneck, SPPF, C2PSA, C3k2, Concat, Detect
from ultralytics.nn.modules.conv import autopad


class BirdShapeAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 16)

        self.h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.h_conv = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
            nn.Conv2d(mid, mid, (1, 7), padding=(0, 3), groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
            nn.Conv2d(mid, channels, 1, bias=True),
        )

        self.v_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.v_conv = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
            nn.Conv2d(mid, mid, (7, 1), padding=(3, 0), groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
            nn.Conv2d(mid, channels, 1, bias=True),
        )

        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, mid, 3, 1, 1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
            nn.Conv2d(mid, channels, 1, bias=True),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = self.h_pool(x)
        x_h = self.h_conv(x_h).expand(-1, -1, h, w)

        x_v = self.v_pool(x)
        x_v = self.v_conv(x_v).expand(-1, -1, h, w)

        x_l = self.local_conv(x)

        fused = self.fusion(torch.cat([x_h, x_v, x_l], dim=1))

        g = self.gate(x)
        return x + fused * g


class C2PSA_BSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c1 = c1
        self.c2 = c2
        self.c = int(c1 * e)
        self.n = n
        self.e = e
        self.m = nn.Sequential(
            C2PSA(c1, c2, n=n, e=e),
            BirdShapeAttention(c1),
        )

    def forward(self, x):
        return self.m(x)


class RGCConv(nn.Module):
    def __init__(self, c1, c2, k=3, groups=4):
        super().__init__()
        self.groups = groups
        self.conv = nn.ModuleList()
        for g in range(groups):
            self.conv.append(
                nn.Sequential(
                    Conv(c1 if g == 0 else c2 // groups, c2 // groups, k, 1, autopad(k, None, 1)),
                )
            )
        self.residual = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        outputs = []
        for i, conv in enumerate(self.conv):
            inp = x if i == 0 else outputs[-1]
            outputs.append(conv(inp))
        out = torch.cat(outputs, dim=1)
        return out + res


class RGC_ELAN(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, groups=4):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        n = max(int(n), 1)
        self.m = nn.Sequential(*(RGCConv(c_, c_, k=3, groups=groups) for _ in range(n)))
        self.cv3 = Conv(c_ * 2, c2, 1, 1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.m(self.cv2(x))
        return self.cv3(torch.cat([y1, y2], dim=1))


class EMA(nn.Module):
    def __init__(self, channels, factor=32):
        super().__init__()
        self.groups = factor
        assert channels // factor > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        gc = channels // factor
        self.gn = nn.GroupNorm(num_groups=gc, num_channels=gc)
        self.conv1x1 = nn.Conv2d(gc, gc, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(gc, gc, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(
            b * self.groups, 1, h, w
        )
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class EMAWrapper(nn.Module):
    def __init__(self, layer, ema):
        super().__init__()
        self.layer = layer
        self.ema = ema

    def forward(self, x):
        return self.ema(self.layer(x))


class TSSABlock(nn.Module):
    def __init__(self, c, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = max(c // num_heads, 1)
        self.ffn = nn.Sequential(Conv(c, c, 1), Conv(c, c, 1))
        self.qkv = Conv(c, c * 3, 1)
        self.proj = Conv(c, c, 1)
        self.spatial_att = nn.Sequential(
            Conv(c, c, 3, 1, 1),
            nn.BatchNorm2d(c),
            nn.Sigmoid(),
        )
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, max(c // 4, 1), 1),
            nn.Conv2d(max(c // 4, 1), c, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
        out = self.proj(out)
        out = out * self.spatial_att(out) * self.channel_att(out)
        return self.ffn(out) + x


class C2TSSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        n = max(int(n), 1)
        self.m = nn.Sequential(*(TSSABlock(self.c) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class DEConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.conv = Conv(c1, c2, k, s)
        self.conv_d1 = nn.Conv2d(c1, c2, k, s, autopad(k), groups=c1, bias=False)
        self.conv_d2 = nn.Conv2d(c1, c2, (k, 1), s, autopad((k, 1)), groups=c1, bias=False)
        self.conv_d3 = nn.Conv2d(c1, c2, (1, k), s, autopad((1, k)), groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self._is_deployed = False

    def forward(self, x):
        if self._is_deployed:
            return self.act(self.bn(self.conv_d1(x)))
        out = self.conv(x)
        d1 = self.conv_d1(x)
        d2 = self.conv_d2(x)
        d3 = self.conv_d3(x)
        return self.act(self.bn(out + d1 + d2 + d3))

    def deploy(self):
        self._is_deployed = True
        fused = self.conv.conv.weight.data + self.conv_d1.weight.data + self.conv_d2.weight.data + self.conv_d3.weight.data
        self.conv_d1 = nn.Conv2d(
            self.conv_d1.in_channels,
            self.conv_d1.out_channels,
            self.conv_d1.kernel_size,
            self.conv_d1.stride,
            self.conv_d1.padding,
            bias=True,
        )
        self.conv_d1.weight.data = fused
        if self.conv.conv.bias is not None:
            self.conv_d1.bias.data = self.conv.conv.bias.data


class EDEC_Detect(Detect):
    def __init__(self, nc=80, ch=()):
        super().__init__(nc=nc, ch=ch)
        c3 = max(ch[0], min(nc, 256))
        self.deconv_stems = nn.ModuleList()
        for c in ch:
            self.deconv_stems.append(
                nn.Sequential(
                    DEConv(c, c, 3, 1),
                    Conv(c, c, 1),
                )
            )

    def forward(self, x):
        enhanced = []
        for i, xi in enumerate(x):
            if i < len(self.deconv_stems):
                enhanced.append(self.deconv_stems[i](xi))
            else:
                enhanced.append(xi)
        return super().forward(enhanced)


class FPSC(nn.Module):
    def __init__(self, c1, c2, k=3, dilations=(1, 3, 6)):
        super().__init__()
        c_ = max(c1 // len(dilations), 16)
        self.cv1 = Conv(c1, c_ * len(dilations), 1, 1)
        self.dw_convs = nn.ModuleList()
        self.dw_bns = nn.ModuleList()
        for d in dilations:
            padding = autopad(k, None, d)
            self.dw_convs.append(
                nn.Conv2d(c_, c_, k, 1, padding, d, groups=c_, bias=False)
            )
            self.dw_bns.append(nn.Sequential(nn.BatchNorm2d(c_), nn.SiLU()))
        self.cv2 = Conv(c_ * len(dilations), c2, 1, 1)

    def forward(self, x):
        y = self.cv1(x)
        chunks = y.split(y.shape[1] // len(self.dw_convs), dim=1)
        results = [bn(conv(ch)) for conv, bn, ch in zip(self.dw_convs, self.dw_bns, chunks)]
        return self.cv2(torch.cat(results, dim=1))


def _get_c3k2_channels(layer):
    c_out = getattr(layer, 'c2', None) or getattr(layer, 'c', None)
    if c_out is None and hasattr(layer, 'cv2') and hasattr(layer.cv2, 'conv'):
        c_out = layer.cv2.conv.out_channels
    c_in = getattr(layer, 'c1', None)
    if c_in is None and hasattr(layer, 'cv1') and hasattr(layer.cv1, 'conv'):
        c_in = layer.cv1.conv.in_channels
    return c_in, c_out


def _get_sppf_channels(layer):
    c_in = layer.cv1.conv.in_channels
    c_out = layer.cv2.conv.out_channels
    return c_in, c_out


def _get_c2psa_channels(layer):
    c = getattr(layer, 'c', None)
    if c is None and hasattr(layer, 'cv1') and hasattr(layer.cv1, 'conv'):
        c = layer.cv1.conv.in_channels
    return c


def _transfer_conv_block(src, dst):
    if hasattr(src, 'conv') and hasattr(dst, 'conv'):
        if src.conv.weight.shape == dst.conv.weight.shape:
            dst.conv.weight.data.copy_(src.conv.weight.data)
            if src.conv.bias is not None and dst.conv.bias is not None:
                dst.conv.bias.data.copy_(src.conv.bias.data)
    if hasattr(src, 'bn') and hasattr(dst, 'bn'):
        if src.bn.weight.shape == dst.bn.weight.shape:
            dst.bn.weight.data.copy_(src.bn.weight.data)
            dst.bn.bias.data.copy_(src.bn.bias.data)
            dst.bn.running_mean.data.copy_(src.bn.running_mean.data)
            dst.bn.running_var.data.copy_(src.bn.running_var.data)


def _transfer_matching_params(src, dst):
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    transferred = 0
    for key in src_sd:
        if key in dst_sd and src_sd[key].shape == dst_sd[key].shape:
            dst_sd[key].copy_(src_sd[key])
            transferred += 1
    dst.load_state_dict(dst_sd)
    return transferred


def _copy_layer_attrs(src, dst):
    for attr in ("f", "i", "type"):
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


def apply_improvements_to_model(detection_model, improvements, nc):
    if not improvements or not any(improvements.values()):
        return

    wrapper = detection_model.model
    total_transferred = 0

    for i, layer in enumerate(wrapper):
        layer_type = type(layer).__name__

        if improvements.get("rgc_elan", False) and layer_type == "C3k2":
            c_in, c_out = _get_c3k2_channels(layer)
            n_blocks = getattr(layer, "n", 1)
            e = max(getattr(layer, "e", 0.25), 0.5)
            new_layer = RGC_ELAN(c_in, c_out, n=n_blocks, e=e)
            n = _transfer_matching_params(layer, new_layer)
            total_transferred += n
            _copy_layer_attrs(layer, new_layer)
            wrapper[i] = new_layer
            print(f"  Layer {i} C3k2->RGC_ELAN: c_in={c_in} c_out={c_out} transferred={n}")

        elif improvements.get("c2tssa", False) and layer_type == "C2PSA":
            c = _get_c2psa_channels(layer)
            n_blocks = getattr(layer, "n", 1)
            e = getattr(layer, "e", 0.5)
            new_layer = C2TSSA(c, c, n=n_blocks, e=e)
            n = _transfer_matching_params(layer, new_layer)
            total_transferred += n
            _copy_layer_attrs(layer, new_layer)
            wrapper[i] = new_layer
            print(f"  Layer {i} C2PSA->C2TSSA: c={c} transferred={n}")

        elif improvements.get("fpsc", False) and layer_type == "SPPF":
            c_in, c_out = _get_sppf_channels(layer)
            new_layer = FPSC(c_in, c_out)
            if hasattr(layer.cv1, 'conv') and hasattr(new_layer.cv1, 'conv'):
                if layer.cv1.conv.weight.shape == new_layer.cv1.conv.weight.shape:
                    _transfer_conv_block(layer.cv1, new_layer.cv1)
                    total_transferred += 1
                    print(f"  Layer {i} SPPF->FPSC: cv1 weights transferred")
            _copy_layer_attrs(layer, new_layer)
            wrapper[i] = new_layer
            print(f"  Layer {i} SPPF->FPSC: c_in={c_in} c_out={c_out}")

    if improvements.get("edec_head", False):
        detect_idx = None
        for i, layer in enumerate(wrapper):
            if isinstance(layer, Detect) and not isinstance(layer, EDEC_Detect):
                detect_idx = i
                break
        if detect_idx is not None:
            old_detect = wrapper[detect_idx]
            ch = tuple(cv[0].conv.in_channels for cv in old_detect.cv2)
            new_detect = EDEC_Detect(nc=nc, ch=ch)
            n = _transfer_matching_params(old_detect, new_detect)
            total_transferred += n
            _copy_layer_attrs(old_detect, new_detect)
            if hasattr(old_detect, 'stride') and hasattr(new_detect, 'stride'):
                new_detect.stride = old_detect.stride.clone()
            if hasattr(old_detect, 'bias_init') and hasattr(new_detect, 'bias_init'):
                new_detect.bias_init = old_detect.bias_init
            print(f"  EDEC_Detect: transferred {n} weight tensors, stride={new_detect.stride}")
            wrapper[detect_idx] = new_detect

    if improvements.get("bsa", False):
        bsa_applied = False
        for i, layer in enumerate(wrapper):
            layer_type = type(layer).__name__
            if layer_type == "C2PSA" and not isinstance(layer, C2PSA_BSA):
                c = getattr(layer, 'c2', None) or getattr(layer, 'c1', None)
                if c is None and hasattr(layer, 'cv2') and hasattr(layer.cv2, 'conv'):
                    c = layer.cv2.conv.out_channels
                if c is not None:
                    n_blocks = getattr(layer, 'n', 1)
                    e = getattr(layer, 'e', 0.5)
                    new_layer = C2PSA_BSA(c, c, n=n_blocks, e=e)
                    n = _transfer_matching_params(layer, new_layer.m[0])
                    total_transferred += n
                    _copy_layer_attrs(layer, new_layer)
                    wrapper[i] = new_layer
                    bsa_applied = True
                    print(f"  Layer {i} C2PSA+BSA: c={c} (bird shape attention)")
                    break
        if not bsa_applied:
            print("  BSA: no C2PSA layer found, skipped")

    if improvements.get("ema", False):
        ema_applied = False
        sppf_or_fpsc_idx = None
        for i, layer in enumerate(wrapper):
            layer_type = type(layer).__name__
            if layer_type in ("SPPF", "FPSC"):
                sppf_or_fpsc_idx = i
                break
        if sppf_or_fpsc_idx is not None:
            for i in range(sppf_or_fpsc_idx + 1, len(wrapper)):
                layer = wrapper[i]
                layer_type = type(layer).__name__
                if layer_type in ("C2PSA", "C3k2", "C2PSA_BSA"):
                    c = getattr(layer, 'c2', None) or getattr(layer, 'c1', None)
                    if c is None and hasattr(layer, 'cv2') and hasattr(layer.cv2, 'conv'):
                        c = layer.cv2.conv.out_channels
                    if c is not None and c % 32 == 0 and c // 32 > 0:
                        ema_mod = EMA(c)
                        wrapper[i] = EMAWrapper(layer, ema_mod)
                        _copy_layer_attrs(layer, wrapper[i])
                        ema_applied = True
                        print(f"  Layer {i} {layer_type}+EMA: c={c} (P5 deep layer)")
                    else:
                        print(f"  Layer {i} {layer_type}: EMA skipped (c={c} not compatible)")
                    break
        if not ema_applied:
            print("  EMA: no suitable P5 layer found, skipped")

    print(f"  Total transferred weight tensors: {total_transferred}")


def build_improved_model(nc, pretrained="yolo11n.pt", improvements=None):
    from ultralytics import YOLO

    if improvements is None:
        improvements = {
            "rgc_elan": True,
            "c2tssa": True,
            "fpsc": True,
            "ema": True,
            "edec_head": True,
        }

    model = YOLO(pretrained)

    if hasattr(model.model, "args") and isinstance(model.model.args, dict):
        model.model.args["nc"] = nc

    model.overrides["nc"] = nc

    apply_improvements_to_model(model.model, improvements, nc)

    return model


def make_improved_trainer(improvements, nc):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class _ImprovedTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            if improvements:
                apply_improvements_to_model(model, improvements, nc)
            return model

    return _ImprovedTrainer





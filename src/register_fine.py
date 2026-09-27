"""Fine 3D registration (similarity: rotation, translation, isotropic scale) of a mid-band
block: 2.4 um level 4 (~38.4 um) against 9.36 um level 2 (~37.4 um).

SimpleITK convention: the transform maps FIXED physical points (9.36 um scan, mm, xyz)
to MOVING physical points (2.4 um scan, mm, xyz). Initialised from <name>_outline.json.
Writes <name>_fine.json with the transform and a held-out slice NCC check.
"""
import json
import sys

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

import zfetch as zf

UM24, UM93 = 2.403, 9.362
VOL = {
    '0846A': ('PHerc0846A/volumes/20260319102732-2.403um-0.2m-77keV-masked.zarr',
              'PHerc0846A/volumes/20250728152254-9.362um-1.2m-113keV-masked.zarr'),
    '1203': ('PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr',
             'PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr'),
}


def img(arr, vox_mm, origin_zyx_mm):
    im = sitk.GetImageFromArray(arr.astype(np.float32))
    im.SetSpacing((vox_mm, vox_mm, vox_mm))
    im.SetOrigin((origin_zyx_mm[2], origin_zyx_mm[1], origin_zyx_mm[0]))
    return im


def main(name, zfrac=0.5, half=64, la=4, lb=2):
    va_url, vb_url = VOL[name]
    coarse = json.load(open(f'{name}_outline.json'))
    va = UM24 * 2 ** la / 1000.0
    vb = UM93 * 2 ** lb / 1000.0
    ma, mb = zf.meta(va_url, la), zf.meta(vb_url, lb)
    za_c = int(ma['shape'][0] * zfrac)
    a = zf.read_box(va_url, la, (za_c - half, 0, 0), (za_c + half, ma['shape'][1], ma['shape'][2]))
    za0_mm = (za_c - half) * va
    zb_c_mm = za_c * va + coarse['z_offset_mm']
    hb = int(half * va / vb) + int(1.0 / vb)
    zb_c = int(round(zb_c_mm / vb))
    b = zf.read_box(vb_url, lb, (zb_c - hb, 0, 0), (zb_c + hb, mb['shape'][1], mb['shape'][2]))
    zb0_mm = (zb_c - hb) * vb
    fixed = img(b, vb, (zb0_mm, 0, 0))
    moving = img(a, va, (za0_mm, 0, 0))
    # initial translation: centroid of material in the central slices, z from coarse offset
    ca = np.array(ndimage.center_of_mass(a[half] > 0)) * va
    cb = np.array(ndimage.center_of_mass(b[hb] > 0)) * vb
    t0 = (ca[1] - cb[1], ca[0] - cb[0], -coarse['z_offset_mm'])   # x, y, z (fixed -> moving)
    tr = sitk.Similarity3DTransform()
    centre = fixed.TransformContinuousIndexToPhysicalPoint([s / 2 for s in fixed.GetSize()])
    tr.SetCenter(centre)
    tr.SetTranslation(t0)
    fmask = sitk.Cast(fixed > 0, sitk.sitkUInt8)
    fmask = sitk.BinaryErode(fmask, [2, 2, 2])
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetMetricFixedMask(fmask)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.05, seed=20260924)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=0.5, minStep=1e-5, numberOfIterations=300,
                                               relaxationFactor=0.6)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SetInitialTransform(tr, inPlace=False)
    final = R.Execute(fixed, moving)
    final = sitk.Similarity3DTransform(sitk.CompositeTransform(final).GetBackTransform()
                                       if isinstance(final, sitk.CompositeTransform) else final)
    # check: NCC of resampled moving vs fixed on material, at level of full block
    res = sitk.Resample(moving, fixed, final, sitk.sitkLinear, 0.0)
    fa, fb = sitk.GetArrayFromImage(res), sitk.GetArrayFromImage(fixed)
    m = (fa > 0) & (fb > 0)
    x, y = fa[m] - fa[m].mean(), fb[m] - fb[m].mean()
    ncc = float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))
    # same with only the initial transform, for reference
    res0 = sitk.Resample(moving, fixed, tr, sitk.sitkLinear, 0.0)
    f0 = sitk.GetArrayFromImage(res0)
    m0 = (f0 > 0) & (fb > 0)
    x0, y0 = f0[m0] - f0[m0].mean(), fb[m0] - fb[m0].mean()
    ncc0 = float((x0 * y0).sum() / np.sqrt((x0 * x0).sum() * (y0 * y0).sum()))
    p = final.GetParameters()
    out = {'name': name, 'zfrac': zfrac, 'levels': [la, lb], 'voxel_mm': [va, vb],
           'final_metric': R.GetMetricValue(), 'stop': R.GetOptimizerStopConditionDescription(),
           'versor_xyz': p[:3], 'translation_xyz_mm': p[3:6], 'scale': p[6],
           'center_xyz_mm': list(final.GetCenter()),
           'rotation_deg_approx': [float(np.degrees(2 * v)) for v in p[:3]],
           'ncc_initial': ncc0, 'ncc_final': ncc, 'matrix': list(final.GetMatrix()),
           'mapping': 'fixed=9.36um physical mm xyz -> moving=2.4um physical mm xyz'}
    print(json.dumps(out, indent=1))
    json.dump(out, open(f'{name}_fine_z{int(zfrac * 100)}.json', 'w'), indent=1)
    np.save(f'{name}_fine_z{int(zfrac * 100)}_check.npy', np.stack([fa[len(fa) // 2], fb[len(fb) // 2]]))
    print('downloaded MB', zf.downloaded() / 1e6)


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.5)

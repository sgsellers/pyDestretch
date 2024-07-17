import pyDestretch as pyd
from astropy.io import fits
import matplotlib.pyplot as plt
import time
import numpy as np

params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 300,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'font.size': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'font.family': 'serif'
}

plt.rcParams.update(params)

rosa_4170_0 = fits.open("test_destretch_data/20211022_151846_4170_kisip.speckle.batch.00.000.final.fits")[0].data
rosa_4170_1 = fits.open("test_destretch_data/20211022_151846_4170_kisip.speckle.batch.00.001.final.fits")[0].data

rosa_gband_0 = fits.open("test_destretch_data/20211022_151846_gband_kisip.speckle.batch.00.000.final.fits")[0].data
rosa_gband_1 = fits.open("test_destretch_data/20211022_151846_gband_kisip.speckle.batch.00.001.final.fits")[0].data

rosa_cak_0 = fits.open("test_destretch_data/20211022_161846_cak_kisip.speckle.batch.00.000.final.fits")[0].data
rosa_cak_1 = fits.open("test_destretch_data/20211022_161846_cak_kisip.speckle.batch.00.001.final.fits")[0].data

zyla_0 = fits.open("test_destretch_data/20211022_151856_halpha_kisip.speckle.batch.00.000.final.fits")[0].data
zyla_1 = fits.open("test_destretch_data/20211022_151856_halpha_kisip.speckle.batch.00.001.final.fits")[0].data

t0 = time.time()
r4170dest = pyd.Destretch(rosa_4170_1, rosa_4170_0, [0, 64, 32, 16], ncores=4, return_vectors=True)
destr4170, vecs = r4170dest.perform_destretch_bspline()
print(time.time() - t0, "seconds to destretch 1k image relative to previous iteration (spline)")

t0 = time.time()
rgbanddest = pyd.Destretch(rosa_gband_1, rosa_gband_0, [0, 64, 32, 16], ncores=4, return_vectors=True)
destrgband, gvecs = rgbanddest.perform_destretch()
print(time.time() - t0, "seconds to destretch 1k image relative to previous iteration (affine)")

np.save("sp_vecs.npy", vecs)
np.save("af_vecs.npy", gvecs)
t0 = time.time()
rcakdest = pyd.Destretch(rosa_cak_1, rosa_cak_0, [0, 64, 32, 16], warp_vectors=vecs, ncores=4)
destrcak = rcakdest.perform_destretch_bspline()
print(time.time() - t0, "seconds to destretch 1k image relative to continuum image")

t0 = time.time()
zyladest = pyd.Destretch(zyla_1, zyla_0, [0, 128, 64, 32], ncores=4)
destr6563 = zyladest.perform_destretch_bspline()
print(time.time() - t0, "seconds to destretch 2k image relative to previous iteration")

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(221)
ax.imshow(rosa_4170_1, vmin=np.nanmin(rosa_4170_0), vmax=np.nanmax(rosa_4170_0))
ax.set_title("Destretch Target")
ax = fig.add_subplot(222)
ax.imshow(rosa_4170_0, vmin=np.nanmin(rosa_4170_0), vmax=np.nanmax(rosa_4170_0))
ax.set_title("Reference Image")
ax = fig.add_subplot(223)
ax.imshow(destr4170, vmin=np.nanmin(rosa_4170_0), vmax=np.nanmax(rosa_4170_0))
ax.set_title("Destretched Image")
ax = fig.add_subplot(224)
ax.imshow(rosa_4170_1/np.nanmean(rosa_4170_1) - destr4170/np.nanmean(destr4170))
ax.set_title("Difference Original-Destr.")
fig.suptitle("Destretch 4170 Relative to Previous Image in Sequence")
fig.savefig("test_destretch_data/4170_dstr.png", bbox_inches='tight')
plt.cla()
plt.close(fig)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(221)
ax.imshow(rosa_gband_1, vmin=np.nanmin(rosa_gband_0), vmax=np.nanmax(rosa_gband_0))
ax.set_title("Destretch Target")
ax = fig.add_subplot(222)
ax.imshow(rosa_gband_0, vmin=np.nanmin(rosa_gband_0), vmax=np.nanmax(rosa_gband_0))
ax.set_title("Reference Image")
ax = fig.add_subplot(223)
ax.imshow(destrgband, vmin=np.nanmin(rosa_gband_0), vmax=np.nanmax(rosa_gband_0))
ax.set_title("Destretched Image")
ax = fig.add_subplot(224)
ax.imshow(rosa_gband_1/np.nanmean(rosa_gband_1) - destrgband/np.nanmean(destrgband), vmin=-0.1, vmax=0.1)
ax.set_title("Difference Original-Destr.")
fig.suptitle("Destretch gband Relative to Previous Image in Sequence")
fig.savefig("test_destretch_data/gband_dstr.png", bbox_inches='tight')
plt.cla()
plt.close(fig)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(221)
ax.imshow(rosa_cak_1, vmin=np.nanmin(rosa_cak_0), vmax=np.nanmax(rosa_cak_0))
ax.set_title("Destretch Target")
ax = fig.add_subplot(222)
ax.imshow(rosa_4170_0, vmin=np.nanmin(rosa_4170_0), vmax=np.nanmax(rosa_4170_0))
ax.set_title("Reference Image")
ax = fig.add_subplot(223)
ax.imshow(destrcak, vmin=np.nanmin(rosa_cak_0), vmax=np.nanmax(rosa_cak_0))
ax.set_title("Destretched Image")
ax = fig.add_subplot(224)
ax.imshow(rosa_cak_1/np.nanmean(rosa_cak_1) - destrcak/np.nanmean(destrcak), vmin=-0.1, vmax=0.1)
ax.set_title("Difference Original-Destr.")
fig.suptitle("Destretch CaK Relative to Continuum Image")
fig.savefig("test_destretch_data/cak_dstr.png", bbox_inches='tight')
plt.cla()
plt.close(fig)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(221)
ax.imshow(zyla_1, vmin=np.nanmin(zyla_0), vmax=np.nanmax(zyla_0))
ax.set_title("Destretch Target")
ax = fig.add_subplot(222)
ax.imshow(zyla_0, vmin=np.nanmin(zyla_0), vmax=np.nanmax(zyla_0))
ax.set_title("Reference Image")
ax = fig.add_subplot(223)
ax.imshow(destr6563, vmin=np.nanmin(zyla_0), vmax=np.nanmax(zyla_0))
ax.set_title("Destretched Image")
ax = fig.add_subplot(224)
ax.imshow(zyla_1/np.nanmean(zyla_1) - destr6563/np.nanmean(destr6563), vmin=-0.1,vmax=0.1)
ax.set_title("Difference Original-Destr.")
fig.suptitle("Destretch 6563 Relative to Previous Image in Sequence")
fig.savefig("test_destretch_data/6563_dstr.png", bbox_inches='tight')
plt.cla()
plt.close(fig)


import pyDestretch as pyd
from astropy.io import fits
import matplotlib.pyplot as plt
import time

rosa_4170_0 = fits.open("test_destretch_data/20211022_151846_4170_kisip.speckle.batch.00.000.final.fits")[0].data
rosa_4170_1 = fits.open("test_destretch_data/20211022_151846_4170_kisip.speckle.batch.00.001.final.fits")[0].data

rosa_gband_0 = fits.open("test_destretch_data/20211022_151846_gband_kisip.speckle.batch.00.000.final.fits")[0].data
rosa_gband_1 = fits.open("test_destretch_data/20211022_151846_gband_kisip.speckle.batch.00.001.final.fits")[0].data

rosa_cak_0 = fits.open("test_destretch_data/20211022_161846_cak_kisip.speckle.batch.00.000.final.fits")[0].data
rosa_cak_1 = fits.open("test_destretch_data/20211022_161846_cak_kisip.speckle.batch.00.001.final.fits")[0].data

zyla_0 = fits.open("test_destretch_data/20211022_151856_halpha_kisip.speckle.batch.00.000.final.fits")[0].data
zyla_1 = fits.open("test_destretch_data/20211022_151856_halpha_kisip.speckle.batch.00.001.final.fits")[0].data

t0 = time.time()
r4170dest = pyd.Destretch(rosa_4170_1, [0, 64, 32, 16], rosa_4170_0, ncores=4, return_vectors=True)
destr4170,vecs = r4170dest.perform_destretch()
print(time.time() - t0, "seconds to destretch 1k image relative to previous iteration")

t0 = time.time()
rcakdest = pyd.Destretch(rosa_cak_1, [0, 64, 32, 16], rosa_cak_0, warp_vectors=vecs, ncores=4)
destrcak = rcakdest.perform_destretch()
print(time.time() - t0, "seconds to destretch 1k image relative to continuum image")

t0 = time.time()
zyladest = pyd.Destretch(zyla_1, [0, 64, 32, 16], zyla_0, ncores=4)
destr6563 = zyladest.perform_destretch()
print(time.time() - t0, "seconds to destretch 2k image relative to previous iteration")

fig = plt.figure()
ax = fig.add_subplot(221)
ax.imshow(rosa_4170_1,origin = 'lower')
ax.set_title("Destretch Target")
ax = fig.add_subplot(222)
ax.imshow(rosa_4170_0,origin = 'lower')
ax.set_title("Reference Image")
ax = fig.add_subplot(223)
ax.imshow(destr4170,origin = 'lower')
ax.set_title("Destretched Image")
ax = fig.add_subplot(224)
ax.imshow(rosa_4170_1 - destr4170,origin = 'lower')
ax.set_title("Difference Original-Destr.")
fig.suptitle("Destretch Relative to Previous Image in Sequence")
fig.savefig("test_destretch_data/4170_dstr.png",bbox_inches = 'tight')
plt.cla()
plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(221)
ax.imshow(rosa_cak_1,origin = 'lower')
ax.set_title("Destretch Target")
ax = fig.add_subplot(222)
ax.imshow(rosa_4170_0,origin = 'lower')
ax.set_title("Reference Image")
ax = fig.add_subplot(223)
ax.imshow(destrcak,origin = 'lower')
ax.set_title("Destretched Image")
ax = fig.add_subplot(224)
ax.imshow(rosa_cak_1 - destrcak,origin = 'lower')
ax.set_title("Difference Original-Destr.")
fig.suptitle("Destretch Relative to Continuum Image")
fig.savefig("test_destretch_data/cak_dstr.png",bbox_inches = 'tight')
plt.cla()
plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(221)
ax.imshow(zyla_1,origin = 'lower')
ax.set_title("Destretch Target")
ax = fig.add_subplot(222)
ax.imshow(zyla_0,origin = 'lower')
ax.set_title("Reference Image")
ax = fig.add_subplot(223)
ax.imshow(destr6563,origin = 'lower')
ax.set_title("Destretched Image")
ax = fig.add_subplot(224)
ax.imshow(zyla_1 - destr6563,origin = 'lower')
ax.set_title("Difference Original-Destr.")
fig.suptitle("Destretch Relative to Previous Image in Sequence")
fig.savefig("test_destretch_data/6563_dstr.png",bbox_inches = 'tight')
plt.cla()
plt.close(fig)


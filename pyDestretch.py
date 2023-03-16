import numpy as np
import scipy.ndimage as scindi
import scipy.signal as scisig
import skimage.transform as sktran

from multiprocessing import Pool
from itertools import repeat


class ToleranceException(Exception):
    """Exception raised for solutions outside the allowed tolerances.

    Attributes:
    -----------
    tolerance -- tolerance value exceeded
    message -- explanation of error
    """

    def __init__(self, tolerance, message="Solution lies outside tolerance range: "):
        self.tolerance = tolerance
        self.message = message
        super().__init__(self.message, self.tolerance)


def _image_align(image, reference, tolerance=None):
    """
    Wraps Scipy's fftconvolve function to return an aligned image to a given reference.
    If the tolerance values are exceeded, attempt instead to align on the inverse image and reference.
    Currently, no derotation is implemented or planned. Get a better telescope if that's an issue for you.

    Parameters:
    -----------
    image : array-like
        2D image to align
    reference : array-like
        2D reference image to align to
    tolerance : None or 2-tuple with (ytolerance,xtolerance).

    Returns:
    --------
    aligned_image : array-like
        Aligned image
    """

    y0 = image.shape[0]/2.
    x0 = image.shape[0]/2.

    y0_ref = reference.shape[0]/2.
    x0_ref = reference.shape[1]/2.

    shifts = np.empty(2)
    aligned_image = np.zeros(image.shape)

    img = image - np.nanmean(image)
    ref = reference - np.nanmean(reference)

    correlation_map = scisig.fftconvolve(img, ref[::-1, ::-1], mode='same')

    y, x = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
    shifts[0] = (y0 - y - (y0 - y0_ref))
    shifts[1] = (x0 - x - (x0 - x0_ref))

    if tolerance is not None:
        if (shifts[0] > tolerance[0]) or (shifts[1] > tolerance[1]):
            im_inv = 1./img
            ref_inv = 1./ref

            corr_map_inv = scisig.fftconvolve(im_inv, ref_inv[::-1, ::-1], mode='same')

            y, x = np.unravel_index(np.argmax(corr_map_inv), corr_map_inv.shape)
            shifts[0] = (y0 - y - (y0 - y0_ref))
            shifts[1] = (x0 - x - (x0 - x0_ref))

            if (shifts[0] > tolerance[0]) or (shifts[1] > tolerance[1]):
                raise ToleranceException(tolerance)

    scindi.shift(image, shifts, output=aligned_image, mode='constant', cval=0.0)

    return aligned_image


def _warp_coords(coordinate, transform_object):
    """Top-level function for performing coordinate warping. Top level for parallelization purposes.

    Parameters:
    -----------
    coordinate : array-like
        Numpy array of shape (N,2) for transformation
    transform : object
        From scikit-image, a geometric transform for application of warp

    Returns:
    --------
    warped_coordinates : array-like
        Numpy array of shape (N,2) of coordinates after transformation
    """
    warped_coordinates = transform_object(coordinate)
    return warped_coordinates


class Destretch:
    """Omnibus class for image destretching. Copied or adapted in large part from Sunspot's image destretch IDL tools
    from the 90's. See original reg.pro for details of individual functions. Now parallelized.

    Attributes:
    -----------
        reference_image -- image used as a reference for calculating destretch vectors
        destretch_target -- image to warp
        target_size -- size of target image
        kernel -- list of kernel sizes
        kernel_size -- kernel size currently in use (as the list is looped over)
    """

    def __init__(self, destretch_target, reference_image, kernel_sizes,
                 warp_vectors=None, ncores=2, return_vectors=False):
        """Initializing the destretch class WITHOUT a reference image.
        The reference image can be provided later, but does not have to be, as the class expects warp_params.

        Parameters:
        -----------
        destretch_target : array-like
            The 2d image to be destretched. Any looping must be done outside the class. I'll provide an example script.
        kernel_sizes : list
            The series of kernel sizes that the destretch is run over.
            A leading zeros indicates that fine aligment should be performed.
        reference_image : array-like
            If warp_vectors is provided, reference_image is used for alignment purposes.
            Else, reference_image is used to calculate warp_vectors.
        warp_vectors : None or array-like
            If an array, assumed to be an arroy of valid warp vectors.
            If none, then reference_image should be provided.
            If both are none, then an error is raised
        return_vectors : bool
            If true, return a list of vector arrays used in the destretch algorithm.
        """
        self.reference_image = reference_image
        self.destretch_target = destretch_target
        self.kernel = kernel_sizes
        self.target_size = self.destretch_target.shape
        if (type(self.kernel) == int) or (type(self.kernel) == float):
            self.kernel_size = self.kernel
        else:
            self.kernel_size = None
        self.ncores = ncores
        self.warp_vectors = warp_vectors
        self.return_vectors = return_vectors
        self.wxy = None
        self.bound_size = None
        self.control_size = None
        self.destretch_image = None

    def perform_destretch(self):
        """Perform image destretch. There are several cases to account for here:
        1.  Reference image and either single kernel size or list thereof are provided.
            In this case, perform iterative destretch as normal.
        2.  Warp vectors are provided.
            In this case, perform just the coordinate remapping using scipy
        """
        if self.warp_vectors is None:
            if self.kernel_size is not None:
                if self.kernel_size == 0:
                    self.destretch_target = _image_align(
                        self.destretch_target,
                        self.reference_image,
                        tolerance=(self.target_size[0]/4., self.target_size[1]/4.))
                else:
                    rcps = self.mkcps()
                    tcps = self.cps(rcps)
                    self.warp_vectors = [self.doreg(rcps, tcps)]
                    self.destretch_image = scindi.map_coordinates(
                        self.destretch_target,
                        self.warp_vectors[0],
                        order=1,
                        mode='constant',
                        prefilter=False,
                        cval=0
                    )
            else:
                self.warp_vectors = []
                for i in range(len(self.kernel)):
                    if self.kernel[i] == 0:
                        self.destretch_target = _image_align(
                            self.destretch_target,
                            self.reference_image,
                            tolerance=(self.target_size[0]/4., self.target_size[1]/4.)
                        )
                    else:
                        self.kernel_size = self.kernel[i]
                        rcps = self.mkcps()
                        tcps = self.cps(rcps)
                        wv = self.doreg(rcps, tcps)
                        self.warp_vectors.append(wv)
                        self.destretch_image = scindi.map_coordinates(
                            self.destretch_target,
                            wv,
                            order=1,
                            mode='constant',
                            prefilter=False,
                            cval=0
                        )
        else:
            if type(self.warp_vectors) is list:
                for i in range(len(self.warp_vectors)):
                    if (self.kernel_size == 0) or (self.kernel[0] == 0):
                        self.destretch_target = _image_align(
                            self.destretch_target,
                            self.reference_image,
                            tolerance=(self.target_size[0]/4., self.target_size[0]/4.)
                        )
                    self.destretch_image = scindi.map_coordinates(
                        self.destretch_target,
                        self.warp_vectors[i],
                        order=1,
                        mode='constant',
                        prefilter=False,
                        cval=0
                    )
            else:
                if (self.kernel_size == 0) or (self.kernel[0] == 0):
                    self.destretch_target = _image_align(
                        self.destretch_target,
                        self.reference_image,
                        tolerance=(self.target_size[0]/4., self.target_size[0]/4.)
                    )
                self.destretch_image = scindi.map_coordinates(
                    self.destretch_target,
                    self.warp_vectors,
                    order=1,
                    mode='constant',
                    prefilter=False,
                    cval=0
                )

        mask = self.destretch_image == 0.
        self.destretch_image[mask] = self.destretch_target[mask]
        if self.return_vectors:
            return self.destretch_image, self.warp_vectors
        else:
            return self.destretch_image

    def mkcps(self):
        """Compute reference image control points given the image size and the size of the desired kernel.

        Parameters:
        -----------
        None additional

        Returns:
        --------
        rcps : array-like
            Array of control points for the reference image. Has the shape (2,npointsx,npointsy).
            This defines a grid in xy corresponding to kernel centers.

        Defines:
        --------
        self.wxy : int
            Wander limits (should be larger than the kernel size)
        self.bound_size : tuple
            A 2-tuple containing the boundary x/y size
        self.control_size : tuple
            A 2-tuple containing control point size
        """

        a = 40./np.sqrt(2.)
        b = 20./np.log(2.)

        self.wxy = int(a*np.exp(-self.kernel_size/b) + self.kernel_size)
        if self.wxy % 2 == 1:
            self.wxy += 1

        cpx = (self.target_size[0] - self.wxy + self.kernel_size)/self.kernel_size
        cpy = (self.target_size[1] - self.wxy + self.kernel_size)/self.kernel_size

        bx = ((self.target_size[0] - self.wxy + self.kernel_size) % self.kernel_size) / 2
        by = ((self.target_size[1] - self.wxy + self.kernel_size) % self.kernel_size) / 2

        rcps = np.zeros((2, int(cpx), int(cpy)))
        
        ly = by
        hy = ly + self.wxy

        for i in range(0, int(cpy)):
            lx = bx
            hx = lx + self.wxy
            for j in range(0, int(cpx)):
                rcps[0, j, i] = (lx + hx) / 2
                rcps[1, j, i] = (ly + hy) / 2
                lx += self.kernel_size
                hx += self.kernel_size
            ly += self.kernel_size
            hy += self.kernel_size
        self.bound_size = (bx, by)
        self.control_size = (cpx, cpy)

        return rcps

    def cps(self, reference_control_points):
        """Compute the offsets for the current image from reference.

        Parameters:
        -----------
        reference_control_points : array-like
            Calculated using self.mkcps(), the control points of the reference image.

        Returns:
        --------
        target_control_points : array-like
            Repaired control points for the target image, corresponding to the same points on the reference grid
        """
        wander_mask = np.ones((self.wxy, self.wxy))
        win = self.doref(wander_mask)
        ans = self.cploc(win, wander_mask)
        target_control_points = self.repair(reference_control_points, ans)
        return target_control_points

    def doref(self, wander_mask):
        """Within the reference window around a control points, normalize the reference window, calculate the complex
        conjugate of the 2d FFT of the window for matching against the same value in the target image.

        Parameters:
        -----------
        wander_mask: array-like
            array of ones of the size of the control point plus border
        Returns:
        --------
        reference_window : array-like
            Complex conjugate of the 2d FFT of the reference image control point window
        """
        reference_window = np.zeros(
            (int(self.wxy),
             int(self.wxy),
             int(self.control_size[0]),
             int(self.control_size[1])),
            dtype=np.csingle)
        ly = int(self.bound_size[1])
        hy = ly + self.wxy
        for j in range(0, int(self.control_size[1])):
            lx = int(self.bound_size[0])
            hx = lx + self.wxy
            for i in range(0, int(self.control_size[0])):
                z = self.reference_image[lx:hx, ly:hy]
                z = z - np.sum(z)/(self.wxy**2)
                reference_window[:, :, i, j] = np.conjugate(np.fft.fft(z * wander_mask))
                lx += self.kernel_size
                hx += self.kernel_size
            ly += self.kernel_size
            hy += self.kernel_size
        return reference_window

    def cploc(self, reference_window, wander_mask):
        """Locate control points on the target image from the reference window

        Parameters:
        -----------
        reference_window : array-like
            Complex array containing the reference image windows from self.doref.
            These are the complex conjugi of the FFT of each window.
        mask : array-like
            Array of ones determining the size of the wander window
        Returns:
        --------
        estimated_target_control_points : array-like
            Estimated locations of control points for the target image. Will require repair for fallacious solutions
        """

        estimated_target_control_points = np.zeros((2, int(self.control_size[0]), int(self.control_size[1])))

        # Setup gradient correction
        # Friedrich Woeger's code

        t1d = (np.arange(0, self.wxy) / (self.wxy - 1) - 0.5) * 2.0
        tx = np.zeros((self.wxy, self.wxy))
        for i in range(tx.shape[0]):
            tx[i, :] = t1d
        ty = np.zeros((self.wxy, self.wxy))
        for i in range(ty.shape[1]):
            ty[:, i] = t1d

        nnx = np.sum(tx * tx)
        nny = np.sum(ty * ty)
        ly = int(self.bound_size[1])
        hy = ly + self.wxy
        for j in range(0, int(self.control_size[1])):
            lx = int(self.bound_size[0])
            hx = lx + self.wxy
            for i in range(0, int(self.control_size[0])):
                ss = self.destretch_target[lx:hx, ly:hy]
                k_tx = np.sum(ss*tx)/nnx
                k_ty = np.sum(ss*ty)/nny
                ss = (ss - (k_tx*tx + k_ty*ty))*wander_mask
                cc = scindi.shift(
                    np.abs(
                        np.fft.fft(
                            np.fft.fft(ss)*reference_window[:, :, i, j]
                        )
                    ),
                    [self.wxy/2, self.wxy/2],
                    mode='grid-wrap'
                )
                mx = np.amax(cc)
                mx_loc = np.array([np.where(cc == mx)[0][0], np.where(cc == mx)[1][0]])
                # Following accounts for fractions of a pixel of shift:
                if (mx_loc[0] * mx_loc[1] > 0) and (mx_loc[0] < (cc.shape[0] - 1)) and (mx_loc[1] < (cc.shape[1] - 1)):
                    denom = mx*2 - cc[mx_loc[0] - 1, mx_loc[1]] - cc[mx_loc[0] + 1, mx_loc[1]]
                    xfra = (mx_loc[0] - 0.5) + (mx - cc[mx_loc[0]-1, mx_loc[1]])/denom

                    denom = mx * 2 - cc[mx_loc[0], mx_loc[1] - 1] - cc[mx_loc[0], mx_loc[1] + 1]
                    yfra = (mx_loc[1] - 0.5) + (mx - cc[mx_loc[0], mx_loc[1] - 1]) / denom

                    mx_loc = np.array([xfra, yfra])

                estimated_target_control_points[0, i, j] = lx + mx_loc[0]
                estimated_target_control_points[1, i, j] = ly + mx_loc[1]

                lx += self.kernel_size
                hx += self.kernel_size
            ly += self.kernel_size
            hy += self.kernel_size
        return estimated_target_control_points

    def repair(self, rcps, tcps, tolerance=0.3):
        """Finds and fixes control points in the target image that drifted too far from the reference grid.
        By "fix" I mean, revert to the reference grid for that point.

        Parameters:
        -----------
        rcps : array-like
            Grid defining the control points on reference image, of size (2,cpx,cpy)
        tcps : array-like
            Should be the same shape as rcps, containing corresponding points in the target image
        tolerance : float, kwarg
            Allowable offset of reference and target control points. In the IDL version, this was hardcoded.
            Here, I'm passing it as a keyword argument. May need to be altered in severe flow environments.

        Returns:
        --------
        target_control_points : array-like
            Edited grid of target control point coordinates.
        """
        kkx = rcps[0, 1, 0] - rcps[0, 0, 0]
        kky = rcps[1, 0, 1] - rcps[1, 0, 0]
        limit = (np.amax([kkx, kky]) * tolerance) ** 2
        difference = tcps - rcps

        bad_points = np.where((difference[0, :, :] ** 2 + difference[1, :, :] ** 2) > limit)

        target_control_points = tcps
        for i in range(len(bad_points[0])):
            target_control_points[:, bad_points[0][i], bad_points[1][i]] = rcps[:, bad_points[0][i], bad_points[1][i]]
        return target_control_points

    def doreg(self, reference_control_points, target_control_points):
        """Do registration and create warp vectors. Parallel workhorse function.
        Uses PiecewiseAffineTransform from scikit-image to warp target image.

        Parameters:
        -----------
        reference_control_points : array-like
            Array of control points on reference image
        target_control_points : array-like
            Array of corresponding locations on target image

        Returns:
        --------
        warp_array : array-like
            Array of warped points in target_image.
        """

        tform = sktran.PiecewiseAffineTransform()
        tform.estimate(
            reference_control_points.reshape(
                (2,
                 reference_control_points.shape[1] * reference_control_points.shape[2])
            ).T,
            target_control_points.reshape(
                (2,
                 target_control_points.shape[1] * target_control_points.shape[2])
            ).T
        )

        warp_array = np.empty((2, self.destretch_target.shape[0], self.destretch_target.shape[1]))

        tf_coords = np.indices((self.destretch_target.shape[1], self.destretch_target.shape[0])).reshape(2, -1).T
        tfc = np.empty(tf_coords.shape)
        tfc_list = []

        nsubfields = 4 * self.ncores

        size_of_subfield = int(tf_coords.shape[0] / nsubfields)

        for i in range(nsubfields):
            if i == (nsubfields - 1):
                tfc_list.append(tf_coords[i*size_of_subfield:, :])
            else:
                tfc_list.append(tf_coords[i*size_of_subfield:(i+1)*size_of_subfield, :])

        with Pool(self.ncores) as p:
            warped_subfields = p.starmap(_warp_coords, zip(tfc_list, repeat(tform)))

        for i in range(len(warped_subfields)):
            if i == len(warped_subfields) - 1:
                tfc[i*size_of_subfield:, :] = warped_subfields[i]
            else:
                tfc[i*size_of_subfield:(i+1)*size_of_subfield, :] = warped_subfields[i]

        tfc = tfc.T.reshape((-1, self.destretch_target.shape[1], self.destretch_target.shape[0])).swapaxes(1, 2)

        warp_array[1, :, :] = tfc[0, :, :]
        warp_array[0, :, :] = tfc[1, :, :]

        return warp_array

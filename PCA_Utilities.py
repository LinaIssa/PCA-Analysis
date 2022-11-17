#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday 13th October, 2022

@author: Lina Issa

A set of useful functions to handle hyperspectral cubes, masks and to perform PCA on them.
"""
# importing stuffs
import numpy  as np

from sklearn.utils.extmath import svd_flip
from sklearn.metrics import mean_squared_error
from scipy import linalg, ndimage
from sklearn.preprocessing import StandardScaler 



################################################################################################
#                                    Functions                                          #
################################################################################################


def resize(datacubes):
    """
    Reshape the image encoded in datacubes into a square image. The shape of the squared image is given by the greatest dimension in datacubes' shape.
    The reshaped data is then filled with zeros to match the shape of a square image.
    This resize operation is needed when regions package is used.
    :param datacubes: hyperspectral data cube nd array 3d
    :return:data hyperspectral data of shape (datacubes.shape[0],l,l) where l is the greatest spatial dimension.
    """
    data = np.zeros((datacubes.shape[0], 259, 259))
    for x in range(datacubes.shape[1]):
        for y in range(datacubes.shape[2]):
            data[:, x, y] = datacubes[:, x, y]
    return data


def plot_spectra(data_cubes, region, mode='avg'):
    """
    Parameters
    ----------
    data_cubes : An hyperspectral cube
    region     : A CirclePixelRegion object that helps extracting the region of interest
    mode       : mode of computing the spectra of the selected region. By default, the extracted spectra is averaged over a spatial slice in the hyperspectral cube

    Returns
    -------
    spectra  : a list of same shape as data_cubes.shape[0]
    """
    bands, pix1, pix2 = data_cubes.shape
    dic = {'avg', 'sum'}
    if mode in dic:
        mask = region.to_mask()

        spectra = []
        if mode == 'avg':
            for l in range(bands):
                myRegion = np.ravel(mask.cutout(data_cubes[l]))
                spectra.append(np.average(myRegion))

        if mode == 'sum':
            for l in range(bands):
                myRegion = np.ravel(mask.cutout(data_cubes[l]))
                spectra.append(np.sum(myRegion))
    else:
        raise TypeError(f'Oups...unrecognised mode. The expected modes are {dic}')

    return spectra


def repliage(bands, pix1, pix2, X, mask):
    """
    Reshape a 2d array into a 3d array.

    Parameters
    ---------
    bands, pix1, pix2 : the shape of the datacube to be constructed
    X    : a 2D array of shape either (number_pixels, bands) or (bands, number_pixels)
    mask : the mask previously applied to get X. TRUE is for BAD pixels

    Returns
    -------
    X_cube : a datacube of shape (bands, pix1, pix2) filled with the values of X and nan otherwise

    """

    x, y = np.arange(pix1), np.arange(pix2)
    MX, MY = np.meshgrid(y, x)
    X_cube = np.full((bands, pix1, pix2), np.nan)
    if bands in X.shape:

        if X.shape[0] == bands:
            for x, y, z in zip(MX[~mask], MY[~mask], range(X.shape[1])):
                X_cube[:, y, x] = X[:, z]
        if X.shape[1] == bands:
            for x, y, z in zip(MX[~mask], MY[~mask], range(X.shape[0])):
                X_cube[:, y, x] = X[z, :]
        return X_cube
    else:
        raise TypeError(
            f'The given number of bands {bands} is in conflict with the shape of the given data {X.shape[0], X.shape[1]}.')


def masked_datacubes(data_cube):
    """
    Returns a 3d masked cube of the same shape as the input data but with a mask
    Calls the function apply_mask to retrieve the good pixels

    Parameters
    -----------
    data_cube : A spectral data cube of shape (number_bands, pix1, pix2)

    Returns
    -------
    masked_cube : ndarray of same shape as the data input.
    The masked values are set to nan

    """
    bands, pix1, pix2 = data_cube.shape

    ############ compute mask & apply it to the data_cube ############

    mask = compute_mask(data_cube)
    masked_data = apply_mask(data_cube, mask)

    ########### Make a 3d cube out of the masked data    ############

    masked_cube = repliage(bands, pix1, pix2, masked_data, mask)
    return masked_cube


def apply_mask(data_cube, mask):
    """
    Applies a mask on the meshgrid formed by the pixels of the spatial dimension of the input data_cube

    Parameters
    ----------
    data_cube: A spectral data cube of shape (number_bands, pix1, pix2)
    mask     : A mask of shape (pix1,pix2) in which TRUE corresponds to the pixels to dispose of and FALSE the pixels to keep   

    Returns
    -------
    returns a 2d array of shape (number of good pixels, number of bands)

    """
    pix1, pix2 = data_cube.shape[1], data_cube.shape[2]
    if mask.shape == (pix1, pix2):
        x, y = np.arange(pix1), np.arange(pix2)
        X, Y = np.meshgrid(y, x)
        output = []
        for x, y in zip(X[~mask], Y[~mask]):
            output.append(data_cube[:, y, x])
        return np.asarray(output)

    else:
        raise TypeError('Wrong mask shape given')


def compute_mask(data_cube):
    """
    Computes a mask in which True is for non-zeros pixels for all the wavelength

    Parameters
    ----------
    data_cube: A spectral data cube of shape (number_bands, pix1, pix2)

    Returns
    -------
    returns the computed mask (TRUE = Bad Pixels)
    """
    mask = np.all(data_cube == 0, axis=0)
    return (mask)


def clipping_datacubes(datacubes, maxiter=5, sigma=3):
    """
    Performs a sigma-clipping over the provided datacubes. Like the function provided by astropy, the data will be iterated over, each time rejecting values that are
    less or more than a specified number of standard deviations from a center values. The center is the median values. The expected data provided is a 3d array

    Clipped pixels are those where:
        |data - median| > sigma * std

    The clipped pixels, instead of being substracted, are substituted with the median values

    Parameters
    ----------
    datacubes : A spectral data cube of shape (number_bands, pix1, pix2)
    maxiter   : Number of iteration. By default 5
    sigma     : By default 3

    Returns
    -------
    new_cube : A spectral cube of same shape as the input cube

    """

    bands, pix1, pix2 = datacubes.shape
    new_cube = np.full((bands, pix1, pix2), np.nan)
    hyper_mask = []
    for l in range(bands):
        image = datacubes[l].copy()
        MasterMask = np.full((image.shape), False)
        for i in range(maxiter):
            median = np.nanmedian(image)  # center values
            diff = image - median
            std = np.nanstd(diff)
            mask = np.abs(diff) > sigma * std
            image[mask] = median  # np.nan
            MasterMask = MasterMask | mask
            if not np.any(mask):
                break
        new_cube[l] = image
        hyper_mask.append(MasterMask)

    return new_cube, np.asarray(hyper_mask)


def MasterMask(HyperMask, mask_allZeros, bands):
    masterMask = mask_allZeros.copy()
    for l in range(bands):
        masterMask = masterMask | HyperMask[l]
    return masterMask


def explained_variance_ratio(eigen_values, n_samples):
    """
    Compute the explained variance ratio which is the percentage of variance explained by each of the features.

    Parameters
    ----------
    eigen_values: list of singular values. MUST BE in a non-increasing order
    n_samples   : number of samples for the training

    Returns
    -------
    explained_variance_ratio; percentage of explained variance

    """
    explained_variance = eigen_values ** 2 / (n_samples - 1)
    total_variance = explained_variance.sum()
    explained_var_ratio = explained_variance / total_variance
    return explained_var_ratio


def pca_nirspec(Yns, nb_comp, masterMask, masked=True):

    """
    @author: Lina Issa

    Performs a PCA on NIRSpec datacubes according to a SVD decomposition.

    returns Z the projection of the cube onto the spectral subspace of
    shape (nb_comp, pix1, pix2) along with V the matrix of projection and
    X_mean the centered deplied data of shape (Lh, pix1 * pix2)
    By default, Z is a masked cube.

    Parameters
    ----------
    Yns     : a spectral cube of shape (Lh, pix1, pix2) in which
    Lh represents the number of spectral bands and pix1
    and pix2 the number of pixels along the 2 spatial axis


    nb_comp : dimension of the spectral subspace, should be
    deduced from the decrease eigenvales plot

    masterMask: a 2d ndarray mask in which True = bad pixels/outliers and False = good pixels

    masking : bool, default = True
    if mask is set to True,  a mask is applied to retrieve
    only the non-zero pixels along the wavelength axis and to remove the outlier via a sigma clipping

    Returns
    -------
    V      : a matrix of projection
    Z_cube : the projected datacube of shape (nb_comp, pix1, pix2)
    X_mean : an ndarray of shape (lh, ). X_mean is the spatial mean of each image of the datacubes

    """
    Lh, pix1, pix2 = Yns.shape

    if masked is None:
        X = np.reshape(Yns.copy(), (Lh, pix1 * pix2)).T  # X depliage de Yns

    else:
        MaskedCubes    = masked_datacubes(Yns)            # removing the all-zeros in the image
        ClippedCube, _ = clipping_datacubes(MaskedCubes) # removing the outliers and replace them with the median
        X              = apply_mask(Yns, masterMask)  # depliage du cube clippé + masqué
        pass

    # PCA

    V, Z, X_mean, S = compute_pca(X, nb_comp)  # Remember Z shape is of shape (number_bands, number_pixels)

    ## Projection du cube

    if masked is None:
        Z_cube = np.reshape(Z.T, (nb_comp, pix1, pix2))  # replie le cube
    else:
        Z_cube = repliage(nb_comp, pix1, pix2, Z, masterMask)
    return V, Z_cube, X_mean, S


def compute_pca(X, nb_comp):
    """
    Performs a SVD decomposition to reduce the spectral dimension

    Parameters
    ----------
    X       : a 2d array of shape (number_pixels, Lh) where Lh is the number of spectral band
    nb_comb : dimension of the spectral subspace, should be deduced from the decrease eigenvales plot

    Returns
    ------
    V      : a ndarray of shape (lh, nb_comp)
    The matrix projection

    Z      : a ndarray of shape (nb_comp, pix1, pix2)
    The projected datacube onto the reduced spectral space

    X_mean : a ndarray of shape (lh, )
    The spatial mean along the number of pixels axis.

    """
    Xpca = X.copy()
    X_mean = np.mean(X, axis=0)  # spectre moyen
    Xpca -= X_mean
    U, S, V = linalg.svd(Xpca, full_matrices=False)

    U, V = svd_flip(U, V)
    S = S[:nb_comp]
    Z = U[:, :nb_comp] * (S ** (1 / 2)).T
    V = np.dot(np.diag(S ** (1 / 2)), V[:nb_comp]).T
    return V, Z, X_mean, S


def retroprojection(V, Z, X_mean, mask, lh):
    """
    Takes the results from the PCA to reconstruct the projected image in the original space.

    Parameters
    ----------
    V : a ndarray of shape (lh, nb_comp).
    It is the matrix of projection as computed by compute_pca.

    Z : a ndarray of shape (nb_comp, pix1, pix2).
    The projected image onto a reduced subspace of dimension nb_components

    X_mean : a ndarray of shape (lh, )

    Returns
    -------
    Z_retro : a 2d ndarray of shape (lh, number of good pixels)
    Z_cube  : a ndarray of shape (lh, pix1, pix2).
    The retroprojected datacube onto the original spectral space
    """
    pix1, pix2 = Z.shape[1], Z.shape[2]
    Z_mask = apply_mask(Z, mask).T
    number_pixels = Z_mask.shape[1]

    Z_retro = np.dot(V, Z_mask) + np.matlib.repmat(X_mean, number_pixels, 1).T  # rétroprojection de Z
    Z_cube = repliage(lh, pix1, pix2, Z_retro, mask)

    return Z_retro, Z_cube

def check_pca(V, Z, X_mean, Yns, mask, method='norm'):
    """
    Checks over the data cube whether the retroprojected one is close to the original data
    Returns the retroprojected data cube along with the error

    Parameters
    ----------

    V      : the matrix of projection calculate in compute_pca
    Z      : the reduced cube to be retroprojected
    X_mean : the the results of compute_pca
    Yns    : the original cube to compare with the retroprojected PCA result Z
    mask   :
    method : str, by default method = 'norm'
    The method to be employed to compute the error. The shape of error will depend on the method


    Returns
    -------
    error  : A quadratic relative error applied when spectral method is employed
    If method is 'norm': returns a float
    If method is 'spectral': returns an array of shape (pix1 *pix2,)
    If method is 'spatial' : returns an arrau of shape (lh, )
    """
    dic = {'norm', 'spectral', 'spatial'}

    pix1, pix2 = Yns.shape[1], Yns.shape[2]
    if method in dic:

        ############### Apply The Mask #####################

        Yns_mask = apply_mask(Yns, mask).T

        bands, number_pixels = Yns_mask.shape[0], Yns_mask.shape[1]
        amplitude = np.nanmean(Yns_mask)

        ############### Retroprojection de Z ###############

        X, X_cube = retroprojection(V, Z, X_mean, mask,
                                    bands)  # np.dot(V, Z_mask)+np.matlib.repmat(X_mean, number_pixels, 1).T  # rétroprojection de Z
        residu = Yns_mask - X

        if method == 'norm':
            error = np.linalg.norm(residu) / (bands * number_pixels)

        if method == 'spectral':  # erreur quadratique + relative
            # error = np.nanmean(residu, axis=0) # retrieve a map of errors
            output = []
            for i in range(number_pixels):
                mse = mean_squared_error(Yns_mask[:, i], X[:, i]) / (amplitude ** 2)
                output.append(mse)
            error = np.asarray(output)

        if method == 'spatial':
            error = np.nanmean(residu, axis=1)

        ############### Repliage de X ###############
        print(bands, pix1, pix2, X.shape, mask.shape)

        return error  # np.reshape(X, (Yns.shape[0], pix1, pix2)),

    else:
        raise ValueError(f'Not the expected method. The expected methods are:{dic}')



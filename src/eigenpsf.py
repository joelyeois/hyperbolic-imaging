"""
@author: Joel Yeo, joelyeo@u.nus.edu
"""

import torch
import time
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from torch.nn.functional import grid_sample
from torchrbf import RBFInterpolator
from tqdm import tqdm


class EigenPSF:
    '''
    A class to create spatially varying blurred images with sampled PSFs
    '''
    def __init__(self, obj, psfs, psf_coord, obj_coord, tile_shape=None, 
                 verbose=True, device='cpu'):
        """
        Parameters
        ----------
        obj : 2D tensor
            The object to be blurred. Should already be zero-padded
            appropriately.
        psfs : 3D tensor, shape: (n, psf.shape[0], psf.shape[1])
            The varying PSFs in list-form. All PSFs must have same shape.
        psf_coord: list of 2 1D tensors
            The corresponding central coordinates for which the PSF is
            associated with in the image. [0]: row coord, [1]: col coord
        obj_coord: list of 2 1D tensors
            The image coordinate grid. [0]: row coord, [1]: col coord
        tile_shape : int or tuple
            If int, implies same no. psfs along horizontal and vertical.
            If tuple, specifies no. of psfs along horizontal and vertical.
        verbose : boolean
            Set to true for print statements. Default: False
        device : str
            Device to compute on. Default: 'cpu';  Option: 'cuda'.
        """
        self.obj = obj.to(device)
        self.psfs = psfs.to(device)
        self.psf_shape = psfs[0].shape
        self.obj_shape = obj.shape
        self.psf_coordr = psf_coord[0].to(device)
        self.psf_coordc = psf_coord[1].to(device)
        self.obj_coordr = obj_coord[0].to(device)
        self.obj_coordc = obj_coord[1].to(device)
        self.n_psfs = torch.tensor(len(psfs))
        self.tile_shape = tile_shape
        self.verbose = verbose
        self.device = device
        if tile_shape is None:
            self.n_blocks = int(np.sqrt(self.n_psfs))
            self.tile_shape = (self.n_blocks, self.n_blocks)

        self.eigenpsfs = None
        self.eigencoeffs = None

    def create_eigenpsfs(self):
        '''
        Creates eigenPSFs using specified methods.

        Parameters
        ----------
        method : str, optional
            'nmf' uses Non-negative Matrix Factorization.
            'pca' uses mean-centered sklearn's PCA, not recommended.
            'eigen' uses eigendecomposition, best for dimensionality reduction.
            'raw' uses 'one-hot-vector'-like encoding.

        Returns
        -------
        None.

        '''
        #timer 
        t0 = time.perf_counter()
        
        #rearrange psfs into a list
        self.flattened_psfs = []
        for i in range(self.n_psfs):
            self.flattened_psfs.append(self.psfs[i].flatten())
        self.flattened_psfs = torch.stack(self.flattened_psfs)
        stacked_psf = self.flattened_psfs
        
        self.weights, self.flattened_eigenpsfs = self.eigpsfs(stacked_psf)
        
        #2D eigenpsf
        self.eigenpsfs = self.flattened_eigenpsfs.reshape(-1, *self.psf_shape)
        
        #timer
        t1 = time.perf_counter()
        if self.verbose:
            print("Eigen PSF computation took %d seconds" %(t1 - t0))

    def eigpsfs(self, psf_stack):
        '''
        Performs linear decomposition using eigendecomposition.

        Parameters
        ----------
        psf_stack : 3D array
            The stack of 2D PSFs, shape of (no. psfs, psf_size, psf_size).

        Returns
        -------
        weights : 2D array
            The weights associated with the PCA basis.
        flattened_eigenpsfs : 2D array
            The PCA basis.
            
        '''
        #build covariance matrix
        cov_matrix = torch.cov(psf_stack)
        self.cov_matrix = cov_matrix
    
        #build eigenvalues and eigenvectors. Each column of eigvec is the vector
        eigval, eigvec = torch.linalg.eigh(cov_matrix)
    
        #sort eigen values and vectors
        sort_arg = torch.flip(torch.argsort(eigval), (0,))
        sorted_eigval = eigval[sort_arg]
        sorted_eigvec = eigvec[:,sort_arg] #Eigenvectors are the columns.
        self.eigenvalues = sorted_eigval
        self.eigenvectors = sorted_eigvec
    
        #eigenpsf
        flattened_eigenpsfs = []
        for i in range(self.n_psfs):
            temp_eigenpsf = 0
            for j in range(self.n_psfs):
    
                #note the [j,i] reversal in indices
                temp_eigenpsf += sorted_eigvec[j,i] * self.psfs[j]#OG
    
            #flattened eigenpsf
            flattened_eigenpsfs.append(temp_eigenpsf.flatten())
        
        #weights
        weights = sorted_eigvec.T
        # weights = flattened_eigenpsfs @ (psf_stack).T
        # self.eigvec = sorted_eigvec
        return weights, torch.stack(flattened_eigenpsfs)

    def view_eigenpsfs(self):
        '''
        Plots the eigen PSFs as a grid.

        Returns
        -------
        None.

        '''
        #using ImageGrid
        fig = plt.figure(dpi = 200)
        grid = ImageGrid(fig,
                         111,
                         nrows_ncols = self.tile_shape)
        for ax, i in zip(grid, range(self.n_psfs)):
            vmax = torch.max(torch.abs(self.eigenpsfs[i]))
            ax.set(xticks = [], yticks = [])
            ax.imshow(self.eigenpsfs[i].cpu(), vmin=-vmax, vmax=vmax, 
                      cmap='twilight')
        plt.show()

    def view_psfs(self):
        '''
        Plots the original PSFs as a grid.

        Returns
        -------
        None.

        '''
        #using ImageGrid
        fig = plt.figure(dpi = 200)
        grid = ImageGrid(fig,
                         111,
                         nrows_ncols = self.tile_shape)

        for ax, i in zip(grid, range(self.n_psfs)):
            vmax = torch.max(torch.abs(self.psfs[i]))
            ax.imshow(self.psfs[i].cpu(), cmap='twilight', vmin=-vmax, vmax=vmax)
            ax.set(xticks = [], yticks = [])
        plt.show()

    def create_eigencoeffs(self, method='rbf', show_progress=True):
        """
        Creates eigen coefficients corresponding to the eigenPSFs.
        (Lauer, 2002, Deconvolution with a spatially variant PSF)
        (Novak, 2021, Imaging through deconvolution with a spatially
         variant point spread function)
        (Turcotte, 2020, Deconvolution for multimode fiber imaging: modeling
         of spatially variant PSF)
        """
        #timer
        t0 = time.perf_counter()

        #interpolation
        self.eigencoeffs = []
        for k in tqdm(range(self.n_psfs), disable=not(show_progress)):
            # eigvec_arr = self.eigvec[:,k]
            eigvec_arr = self.weights[k,:]
            reshaped_eigvec_arr = eigvec_arr.reshape(self.tile_shape)
            interp_eigvec = self.interpolate2D(reshaped_eigvec_arr,
                                               method=method)
            self.eigencoeffs.append(interp_eigvec)

        self.eigencoeffs = torch.stack(self.eigencoeffs)
        #end timer
        t1 = time.perf_counter()
        if self.verbose:
            print("Eigen coefficients computation took %d seconds" %(t1 - t0))

    def create_eigencoeffs_nongridded(self):
        '''
        Creates nongridded eigen coefficients via RBF interpolation based on
        the decomposition weights.

        Returns
        -------
        None.

        '''
        #timer
        t0 = time.perf_counter()

        #interpolation
        self.eigencoeffs = []
        for k in range(self.n_psfs):
            eigvec_arr = self.weights[k,:]
            
            #use RBF interpolator for non-gridded input. Transpose because
            #interpolator takes in (N,2) input.
            points = torch.stack([self.psf_coordr, self.psf_coordc], dim=1)
            interpolator = RBFInterpolator(points.type(torch.float32),
                                           eigvec_arr.reshape(-1).type(torch.float32),
                                           smoothing=0,
                                           kernel='thin_plate_spline',
                                           device=self.device)
            obj_points = torch.meshgrid(self.obj_coordr,
                                        self.obj_coordc,
                                        indexing='xy')
            obj_points = torch.stack(obj_points, dim=-1).reshape(-1, 2)
            interp_vals = interpolator(obj_points.type(torch.float32))
            #maybe need to flipup to take care of reversed order of yaxis in meshgrid)
            # self.eigencoeffs.append(torch.flipud(interp_vals.reshape(self.obj_shape)))
            self.eigencoeffs.append(interp_vals.reshape(self.obj_shape))
            
        #end timer
        t1 = time.perf_counter()
        if self.verbose:
            print("Eigen coefficients computation took %d seconds" %(t1 - t0))

    def interpolate2D(self, input, method='rbf'):
        sh = input.shape
        if method == 'grid_sample':
            #obtain normalized object coordinates
            xit, yit = utils.normalize_coordinates(self.psf_coordr,
                                                self.psf_coordc,
                                                self.obj_coordr,
                                                self.obj_coordc)
            output = utils.torch_interpolate2D(input,
                                            yit, xit,
                                            mode='bicubic',
                                            padding_mode='zeros',
                                            align_corners=True)
        elif method == 'rbf':
            psf_points = torch.meshgrid(self.psf_coordr,
                                        self.psf_coordc,
                                        indexing='xy')
            psf_points = torch.stack(psf_points, dim=-1).reshape(-1, 2)
            interpolator = RBFInterpolator(psf_points.type(torch.float32),
                                           input.reshape(-1).type(torch.float32),
                                           smoothing=0,
                                           kernel='thin_plate_spline',
                                           device=self.device)
            obj_points = torch.meshgrid(self.obj_coordr,
                                        self.obj_coordc,
                                        indexing='xy')
            obj_points = torch.stack(obj_points, dim=-1).reshape(-1, 2)
            interp_vals = interpolator(obj_points.type(torch.float32))
            output = interp_vals.reshape(self.obj_shape)
        return output
        
    def view_eigencoeffs(self):
        '''
        Plots the interpolated eigencoefficients on a grid.

        Returns
        -------
        None.

        '''
        #using ImageGrid
        fig = plt.figure(dpi = 200)
        grid = ImageGrid(fig,
                         111,
                         nrows_ncols = self.tile_shape)

        vmax = torch.max(torch.abs(self.eigencoeffs))
        for ax, i in zip(grid, range(self.n_psfs)):
            ax.set(xticks = [], yticks = [])
            ax.imshow(self.eigencoeffs[i].cpu(),
                      cmap = 'twilight',
                      vmax = vmax,
                      vmin = -vmax)

    def eigen_convolution(self, obj=None, n_components=None):
        """
        Uses eigen PSFs and coefficients to spatially convolve the object.
        (Lauer, 2002, Deconvolution with a spatially variant PSF)
        (Novak, 2021, Imaging through deconvolution with a spatially
         variant point spread function)
        (Turcotte, 2020, Deconvolution for multimode fiber imaging: modeling
         of spatially variant PSF)
        """
        #timer
        t0 = time.perf_counter()
        
        if obj is None:
            obj = self.obj
        
        img = 0
        self.eigenimg = []
        
        #number of components to use
        if n_components is None:
            n_components = self.n_psfs
            
        for i in range(n_components):
            tmp_eigenimg = utils.fftconvolve(obj * self.eigencoeffs[i],
                                          self.eigenpsfs[i], mode='same')
            self.eigenimg.append(tmp_eigenimg)
            img += tmp_eigenimg

        self.eigenimg = torch.stack(self.eigenimg)
        self.img = img
        
        #end timer
        t1 = time.perf_counter()
        if self.verbose:
            print("Eigen convolution took %d seconds" %(t1 - t0))
        return img

    def view_image(self):
        plt.figure(dpi = 100)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.img.cpu())
        plt.colorbar()
        plt.show()
        
    def normalizing_mask(self, n_components=None):
        '''
        Computes a normalizing mask to remove hot spots. Assumes that a flat
        object will remain flat after blurring which is false. This is merely
        used as a duct-tape solution, not to be taken as physically correct.

        Returns
        -------
        mask : 2D array
            The 2D mask with same shape as the object.

        '''
        #number of components to use
        if n_components is None:
            n_components = self.n_psfs
        
        self.mask = self.eigen_convolution(obj=torch.ones(self.obj_shape, device=self.device),
                                      n_components=n_components)
        
    def blur_image(self, plots=True, normalize=False, n_components=None,
                  method='rbf'):
        #perform linear decomposition
        if self.eigenpsfs is None:
            self.create_eigenpsfs()
        if self.eigencoeffs is None:
            self.create_eigencoeffs(method=method)

        #number of components to use
        if n_components is None:
            n_components = self.n_psfs
        
        #normalization
        if normalize:
            self.mask = self.normalizing_mask(n_components=n_components)
            img = self.eigen_convolution(n_components=n_components)
            self.img = img / self.mask
        else:
            self.img = self.eigen_convolution(n_components=n_components)
        
        #plots
        if plots:
            if self.verbose:
                print('Plotting...')
            self.view_psfs()
            self.view_eigenpsfs()
            self.view_eigencoeffs()
            self.view_image()
"""
EigenPSF algorithm.
@author: Joel Yeo, joelyeo@u.nus.edu
"""

import torch
from . import utils
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from torchrbf import RBFInterpolator
from tqdm import tqdm

class EigenPSF:
    '''
    A class to create spatially varying blurred images with sampled PSFs.
    '''
    def __init__(self, obj, psfs, psf_coord, obj_coord, tile_shape=None, device='cpu'):
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
        self.device = device
        if tile_shape is None:
            self.n_blocks = int(np.sqrt(self.n_psfs))
            self.tile_shape = (self.n_blocks, self.n_blocks)

        self.eigenpsfs = None
        self.eigencoeffs = None

    def create_eigenpsfs(self):
        '''Creates eigenPSFs'''
        
        #rearrange psfs into a list
        self.flattened_psfs = []
        for i in range(self.n_psfs):
            self.flattened_psfs.append(self.psfs[i].flatten())
        self.flattened_psfs = torch.stack(self.flattened_psfs)
        stacked_psf = self.flattened_psfs
        
        
        self.weights, self.flattened_eigenpsfs = self.eigpsfs(stacked_psf) 
        
        #2D eigenpsf
        self.eigenpsfs = self.flattened_eigenpsfs.reshape(-1, *self.psf_shape)
        

    def eigpsfs(self, psf_stack):
        '''
        Performs linear decomposition using eigendecomposition.

        Parameters
        ----------
        psf_stack : 3D tensor
            The stack of 2D PSFs, shape of (no. psfs, psf_size, psf_size).

        Returns
        -------
        weights : 2D tensor
            The weights associated with the PCA basis.
        flattened_eigenpsfs : 2D tensor
            The unraveled eigenpsf basis.
            
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
        '''Plots the eigen PSFs as a grid.'''
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
        '''Plots the original PSFs as a grid.'''
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

        References
        ----------
        (Lauer, 2002, Deconvolution with a spatially variant PSF)
        (Novak, 2021, Imaging through deconvolution with a spatially
         variant point spread function)
        (Turcotte, 2020, Deconvolution for multimode fiber imaging: modeling
         of spatially variant PSF)
        """
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

    def interpolate2D(self, input_arr, method='rbf'):
        '''
        2D Interpolation methods.

        Parameters
        ----------
        input_arr : 2D tensor
            The 2D tensor to be interpolated. The pixels in this 2D tensor have coordinates
            given by the meshgrid of [self.psf_coordr, self.psf_coordc]. The query
            points to be interpolated is the meshgrid of [self.obj_coordr, self.obj_coordc].
        method : str
            The interpolation method. Either 'rbf' or 'grid_sample'.

        Returns
        -------
        output : 2D tensor
            The interpolated 2D tensor.
            
        '''
        sh = input_arr.shape
        if method == 'grid_sample':
            #obtain normalized object coordinates
            xit, yit = utils.normalize_coordinates(self.psf_coordr,
                                                self.psf_coordc,
                                                self.obj_coordr,
                                                self.obj_coordc)
            output = utils.torch_interpolate2D(input_arr,
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
                                           input_arr.reshape(-1).type(torch.float32),
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
        '''Plots the interpolated eigencoefficients on a grid.'''
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
        '''
        Uses eigen PSFs and coefficients to spatially convolve the object.

        Parameters
        ----------
        obj : 2D tensor (optional)
            The 2D object to be spatially-varying blurred. Default is the original object.
        n_components : int (optional)
            The number of eigenpsfs to use for convolution. Default uses all.

        Returns
        -------
        img : 2D tensor
            The blurred object.
            
        '''
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
        
        return img

    def view_image(self):
        '''Plots the blurred object.'''
        plt.figure(dpi = 100)
        plt.xticks([])
        plt.yticks([])
        vmax = torch.max(self.img.cpu())
        plt.imshow(self.img.cpu(), vmin=-vmax, vmax=vmax, cmap='twilight')
        plt.colorbar()
        plt.show()
        
    def blur_image(self, plots=True, n_components=None, method='rbf'):
        '''
        Wrapper function to create eigenPSFs and blur the image.
        
        Parameters
        ----------
        plots : str (optional)
            Set to True to output all plots.
        n_components : int (optional)
            The number of eigenpsfs to use for convolution. Default uses all.
        method : str
            The interpolation method. Either 'rbf' or 'grid_sample'.

        Returns
        -------
        None
        '''
        #perform linear decomposition
        if self.eigenpsfs is None:
            self.create_eigenpsfs()
        if self.eigencoeffs is None:
            self.create_eigencoeffs(method=method)

        #number of components to use
        if n_components is None:
            n_components = self.n_psfs
        
        self.img = self.eigen_convolution(n_components=n_components)
        
        #plots
        if plots:
            self.view_psfs()
            self.view_eigenpsfs()
            self.view_eigencoeffs()
            self.view_image()
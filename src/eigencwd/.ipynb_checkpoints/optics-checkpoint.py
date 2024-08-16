"""
Optics toolbox.
@author: Joel Yeo, joelyeo@u.nus.edu
"""

import torch
from . import torch_propagators as tp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from . import utils
from tqdm import tqdm
import scipy.ndimage as ndimg

torch.set_default_dtype(torch.float64)

fft2 = lambda array: torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(array)))
ifft2 = lambda array: torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(array)))
class lens:
    """
    A class to create a circular lens. Also simulates its on and off-axis PSFs.
    """
    def __init__(self, diameter=500e-6, lenswavelength=740e-9, N=2000, 
                 f=173e-6, dx=None, device=None):
        """
        Initializes lens grid and general lens parameters.
        
        Parameters
        ----------
        diameter: float, optional
            Diameter of lens
        lenswavelength: float, optional
            Central wavelength of lens to create phase profile in [m]
        N: int, optional
            Number of pixels on lens grid. An odd number is preferable for 
            symmetry
        dx: float, optional
            Spatial pixel spacing on lens grid in [m]
        """
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device
        self.diameter = diameter
        self.lens_wavelength = lenswavelength
        self.f = f
        self.N = N
        if dx is None:
            #set to Nyquist sampling limit
            self.dx = lenswavelength/2
        else:            
            self.dx = dx
        self.pupil_function = torch.from_numpy(utils.circle2d(self.N, self.diameter // self.dx).astype(np.float32))

        #lens parameters
        self.NA = ((self.diameter / 2)
                    / torch.sqrt(torch.tensor((self.diameter / 2)**2 + self.f**2)))
        self.abbe_reso_lim = self.lens_wavelength / 2 / self.NA
        self.angular_reso = 1.22 * self.lens_wavelength / self.diameter
        self.f_number = self.f / self.diameter

        #initialize coordinate grids
        self.dy = dx
        self.x = torch.linspace(-0.5, 0.5, self.N) * self.dx * self.N
        self.y = self.x
        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing='xy')
        
        # initialize None object
        self.objectx = None
        self.objecty = None
        self.object = None

    def hyperbolic_phase_profile(self):
        """Returns hyperbolic phase profile"""
        
        R2 = self.X**2 + self.Y**2
        phase = (2 * torch.pi / self.lens_wavelength  * self.pupil_function
                 * (self.f - torch.sqrt(R2  + self.f**2))) 
        return phase
    
    def parabolic_phase_profile(self):
        """Returns parabolic phase profile"""
        
        R2 = self.X**2 + self.Y**2
        phase = (-torch.pi / self.lens_wavelength / self.f * R2 
                 * self.pupil_function)
        return phase

    def spherical_phase_profile(self):
        """Returns spherical phase profile"""
        
        R2 = self.X**2 + self.Y**2
        phase = (2 * torch.pi / self.lens_wavelength * self.pupil_function 
                 * (torch.sqrt(torch.abs(self.f**2 - R2)) - self.f))
        return phase
    

    def create_lens_function(self, custom_lens_phase_profile=None,
                             custom_lens_function=None, 
                             lens_type='hyperbolic'):
        """
        Creates the lens complex transmission function.

        Parameters
        ----------
        custom_lens_phase_profile : ndarray, 2D, optional
            User-defined phase profile for the lens. Assumes standard pupil
            function defined in __init__. The default is None.
        custom_lens_function : TYPE, optional
            User-defined complex lens transmission function. The default is 
            None.
        lens_type : TYPE, optional
            Specify the type of phase profile of lens. The default is 
            'hyperbolic'.

        Returns
        -------
        lens_function : ndarray, 2D
            The lens complex transmission function.

        """
        
        self.lens_type = lens_type
        
        #create standard lens phase profile unless user specifies custom phase
        #profile
        if custom_lens_phase_profile is not None:
            self.lens_phase_profile = custom_lens_phase_profile
        elif self.lens_type == 'hyperbolic':
            self.lens_phase_profile = self.hyperbolic_phase_profile()
        elif self.lens_type == 'parabolic':
            self.lens_phase_profile = self.parabolic_phase_profile()
        elif self.lens_type == 'spherical':
            self.lens_phase_profile = self.spherical_phase_profile()
        elif self.lens_type == 'axicon':
            self.lens_phase_profile = self.axicon_phase_profile()
                    
        #create complex lens function unless user specifies custom lens
        #function
        if custom_lens_function is not None:
            self.lens_function = custom_lens_function
        else:            
            self.lens_function = (torch.exp(1j * self.lens_phase_profile) 
                                  * self.pupil_function)
        return self.lens_function

    def view_lens_function(self, units_scale=1e-3):
        """Plots the complex lens function and its fourier transform"""
        
        fig, axes = plt.subplots(nrows=2, ncols=2, dpi=200, sharey='row', 
                                 sharex=False, constrained_layout=True,
                                 figsize=(7,7))
        
        #spatial units
        x = self.x / units_scale
        y = self.y / units_scale
        if units_scale == 1e-3:
            unit = '(mm)'
        elif units_scale == 1e-6:
            unit = r'($\mu$m)'
        elif units_scale == 1e-9:
            unit = '(nm)'
        else:
            unit = f'($\\times$ {units_scale * 1e3: .0e} mm)'
        
        #frequency units
        kx = torch.fft.fftshift(torch.fft.fftfreq(len(x), x[1] - x[0], device='cpu'))
        ky = torch.fft.fftshift(torch.fft.fftfreq(len(y), y[1] - y[0], device='cpu'))
        if units_scale == 1e-3:
            kunit = r'(mm$^{-1}$)'
        elif units_scale == 1e-6:
            kunit = r'($\mu$m$^{-1}$)'
        elif units_scale == 1e-9:
            kunit = r'(nm$^{-1}$)'
        else:
            kunit = f'($\\times$ {units_scale * 1e3: .0e} mm)$^{-1}$'
            
        #amplitude and phase of lens function
        ax = axes[0,0]
        ax.imshow(torch.abs(self.lens_function.cpu()),
                  extent = [x[0].cpu(), x[-1].cpu(), y[0].cpu(), y[-1].cpu()])
        ax.set(xlabel=' '.join([r'$x$', unit]), 
               ylabel=' '.join([r'$y$', unit]),
               title='amplitude of $f_\mathrm{lens}$')
        
        ax = axes[0,1]
        ax.imshow(torch.angle(self.lens_function.cpu()), cmap='twilight',
                  extent = [x[0].cpu(), x[-1].cpu(), y[0].cpu(), y[-1].cpu()])
        ax.set(xlabel=' '.join([r'$x$', unit]),
               title='phase of $f_\mathrm{lens}$')
        
        #amplitude and phase of fft2 of lens function
        ff2_lens_function = fft2(self.lens_function)
        ax = axes[1,0]
        ax.imshow(torch.abs(ff2_lens_function.cpu()),
                  extent = [kx[0], kx[-1], ky[0], ky[-1]])
        ax.set(xlabel=' '.join([r'$k_x$', kunit]), 
               ylabel=' '.join([r'$k_y$', kunit]),
               title=r'amplitude of $\mathcal{F}\left\{f_\mathrm{lens}\right\}$')
        
        ax = axes[1,1]
        ax.imshow(torch.angle(ff2_lens_function.cpu()), cmap='twilight',
                  extent = [kx[0], kx[-1], ky[0], ky[-1]])
        ax.set(xlabel=' '.join([r'$k_x$', kunit]),
               title=r'phase of $\mathcal{F}\left\{f_\mathrm{lens}\right\}$')
        
        plt.show()

    def create_object(self, obj=None, object_distance=1, object_size=1,
                      max_half_angle_fov=None):
        """
        Creates an object on a coordinate grid using user-input image.

        Parameters
        ----------
        obj: ndarray, 2D, optional
            A 2D image in ndarray form. Best to be odd-sized for symmetry
        object_distance: float, optional
            Distance between object and lens [m]
        max_half_angle_fov: float, optional
            The maximum allowed half angle FOV of the lens. Determines maximum 
            object_length if specified
        object_size: float or tuple, optional
            If float, assume object is square with specified side lengths.
            If tuple, (height, width) of the object [m]
        """
        self.object = obj
        self.objectdistance = object_distance
        self.max_half_angle_fov = max_half_angle_fov
        self.max_object_length = None
        
        if self.max_half_angle_fov is not None:
            self.max_object_length = (2 * self.objectdistance
                                     * torch.tan(self.max_half_angle_fov))
            
        if isinstance(object_size, tuple):
            self.object_size = object_size
        else:
            self.object_size = (object_size, object_size)
            
        # object coordinates
        self.objectx = torch.linspace(-self.object_size[1]/2, self.object_size[1]/2, obj.shape[1])
        self.objecty = torch.linspace(-self.object_size[0]/2, self.object_size[0]/2, obj.shape[0])
        
        if self.max_object_length is not None:
            if max(self.object_size) > self.max_object_length:
                print('Created object is larger than maximum angular FOV')
        self.lens_mag = self.f / self.objectdistance
        self.lens_cutoff_frequency = (self.diameter
                                     / 2
                                     / self.lens_wavelength
                                     / self.objectdistance)
        self.spatial_reso = self.angular_reso * self.objectdistance
        self.half_angle_fov = torch.arctan(torch.tensor(max(self.object_size) / 2 
                                       / self.objectdistance)) / torch.pi * 180
    
    def create_psf(self, wavelength, distance, psfmag=1, pos=(0,0),
                   propagator='scaled blas', normalize=True, prop_dist=None,
                   shiftpsf=True, debarrel=False, source='plane',
                  propagation_medium_refractive_index=1):
        """
        Creates the complex point spread function based on a point source.

        Parameters
        ----------
        wavelength : float
            Wavelength of point source [m]
        distance : float, optional
            z-distance between point source and lens [m]
        psfmag : float, optional
            Objective magnification to magnify the PSF grid. Need to check for 
            aliasing manually. Set to 1 to have PSF grid same as lens grid.
        pos : tuple, optional
            (x,y) coordinate of the point source.
        propagator: str, optional
            Type of propagator used. Check propagators.py for available 
            methods and details.
        normalize : boolean, optional
            If True, normalizes PSF such that its sum is 1. 
        prop_dist : float, optional
            The distance between the lens and camera. If None, assume same as
            focal length of lens: self.f.
        shiftpsf : boolean, optional
            If False, the PSF 2D matrix/image is centered at the origin of the 
            lens' xy-coordinate. If True, the PSF is centered instead on the
            position of where the chief ray emitted from the point source
            hits the camera plane.
        debarrel : boolean
            Removes barrel distortion from the resultant PSF image. shiftpsf 
            must be set to False. Debarrelling also crops the image size to 
            max of 2f x 2f.
        """
        self.source_wavelength = wavelength
        self.source_distance = distance
        self.source_pos = pos
        self.psf_mag = psfmag
        self.propagator = propagator
        if prop_dist is None:
            self.prop_dist = self.f
        else:
            self.prop_dist = prop_dist
        #definition of linear magnification of thin lens, M = -d_image/d_obj
        self.lens_mag = self.prop_dist / distance

        if source == 'point':
            #create spherical wave originating from coordinate: pos = (x,y)
            r =  torch.sqrt((self.X - pos[0])**2 + (self.Y - pos[1])**2
                         + distance**2)
            source_wave = torch.exp(1j * 2 * torch.pi / wavelength * r) / r
        elif source == 'plane':
            k = 2 * torch.pi / wavelength
            theta_x = torch.arctan(-pos[0] / distance)
            theta_y = torch.arctan(-pos[1] / distance)
            source_wave = torch.exp(1j * k * (self.X * torch.sin(theta_x) 
                                 + self.Y * torch.sin(theta_y)))

        #multiply with lens function
        afterlens = (torch.abs(source_wave) * self.lens_function 
                     * torch.exp(1j * torch.angle(source_wave) 
                              * self.pupil_function))

        #propagate transmitted field to camera plane using scaled BLAS method
        if propagator == 'scaled blas':
            psf, psf_x, psf_y = tp.scaledblas(afterlens.to(self.device),
                                              self.prop_dist,
                                              wavelength/propagation_medium_refractive_index, 
                                              self.dx,
                                              self.dx / self.psf_mag)
        #bring back to cpu
        # psf, psf_x, psf_y = psf.to('cpu'), psf_x.to('cpu'), psf_y.to('cpu')
        
        #shift pixels to center. If off-axis PSFs don't look centered, implies 
        #that lens causes geometrical distortions.
        y_shift = pos[1] * self.lens_mag * self.psf_mag / self.dx
        x_shift = pos[0] * self.lens_mag * self.psf_mag / self.dx
        shiftedpsf = utils.torch_shift(psf, (y_shift, x_shift),
                                    padding_mode='zeros')
        #normalize psf
        if normalize:
            shiftedpsf = shiftedpsf / torch.sum(shiftedpsf)
            psf = psf / torch.sum(psf)
        
        if shiftpsf:
            return shiftedpsf
        else:
            if debarrel:
                #crop psf window to max 2f x 2f, else coordinate transform 
                #fails.
                maxnum = torch.tensor(2 * self.f / self.dx)
                maxnumodd = int(torch.floor(maxnum) // 2 * 2 - 1)
                cropped_psf = utils.crop(torch.abs(psf), maxnumodd)
                
                #required inputs for Anton's debarrel function
                ratio = self.dx
                f = self.f
                f_px = f / ratio # in pixel
                psf = remove_barrel(cropped_psf, f_px, x_shift, y_shift)
                
            return psf

    def create_varying_psfs(self, n_blocks=5, psf_size=151, psfmag=2,
                            wavelength=None, normalize=True,
                            extend_to_edge=False, prop_dist=None,
                            debarrel=False, source='plane',
                            propagation_medium_refractive_index=1):
        '''
        Creates tiled array of 2D PSFs corresponding to the location of grid of
        evenly spaced point sources at the object plane.

        Parameters
        ----------
        n_blocks : int or tuple
            If int: number of tiles in one axis, produces a square grid of 
            tiles. Minimum of 5 required.
            If tuple: contains number of tiles in row x col, eg. (row, col).
            Rectangular grid of tiles allowed.
        psf_size: int
            Size of PSF window (psf_size, psf_size) in pixels. Cannot be
            larger than self.N (number of pixels in lens grid).
        psfmag: float
            Objective magnification to magnify the PSF grid. Need to check for
            aliasing manually if set larger than 1.
        wavelength : float
            Wavelength of the point souces.
        nthreads : int
            Number of computing threads to use for multithreading. If not
            specified, uses all threads available (might cause PC lag).
        normalize : boolean
            If True, normalizes each simulated PSF sum to 1.
        extend_to_edge : boolean
            If True, point sources are evenly spaced from the edges of the 
            simulation window. If False, point sources are located at the 
            center of the tiles.
        prop_dist : float
            The distance between the lens and camera. If None, assume same as
            focal length of lens.
        debarrel : boolean
            Removes barrel distortion from off-axis PSFs. shiftpsf must be 
            False. Debarrelling also crops the image size to max of 2f x 2f.
            Setting this to True might be slow, make sure you know what you
            are doing.
        '''
        
        #ensures correct shiftpsf setting.
        if debarrel:
            shiftpsf = False
        else:
            shiftpsf = True
        
        #use len's central wavelength for point sources if not specified
        if wavelength is None:
            wavelength = self.lens_wavelength

        #if user specifies same shape in both axis, use optimized version
        #for square grid.
        if isinstance(n_blocks, tuple):
            if n_blocks[0] == n_blocks[1]:
                n_blocks = n_blocks[0]

        #4D-array containing the 2D PSFs and the position of the tiles
        if isinstance(n_blocks, int):
            psf = torch.zeros((n_blocks, n_blocks, psf_size, psf_size))
            #store the grid shape of PSFs
            self.tile_shape = (n_blocks, n_blocks)
        elif isinstance(n_blocks, tuple):
            psf = torch.zeros(n_blocks + (psf_size, psf_size))
            #store the grid shape of PSFs
            self.tile_shape = n_blocks
            
        #determine coordinates of all point sources using object plane 
        #coordinates
        if extend_to_edge:
            #evenly spaced point sources from edge to edge
            if isinstance(n_blocks, int):
                blocks_central_row_coord = (torch.linspace(-0.5, 0.5, n_blocks) 
                                            * self.object_size[0])
                
                blocks_central_col_coord = (torch.linspace(-0.5, 0.5, n_blocks) 
                                            * self.object_size[1])
                
            elif isinstance(n_blocks, tuple):
                blocks_central_row_coord = (torch.linspace(-0.5, 0.5, n_blocks[0]) 
                                            * self.object_size[0])
                
                blocks_central_col_coord = (torch.linspace(-0.5, 0.5, n_blocks[1]) 
                                            * self.object_size[1])
                
        else:
            #coordinates of central pixels in each tile
            if isinstance(n_blocks, int):
                blocks_central_row_coord = ((torch.arange(n_blocks) 
                                             - n_blocks // 2)
                                            * self.object_size[0] / n_blocks)
                
                blocks_central_col_coord = ((torch.arange(n_blocks) 
                                             - n_blocks // 2)
                                            * self.object_size[1] / n_blocks)
                
            elif isinstance(n_blocks, tuple):
                blocks_central_row_coord = ((torch.arange(n_blocks[0]) 
                                             - n_blocks[0] // 2)
                                            * self.object_size[0] 
                                            / n_blocks[0])
                
                blocks_central_col_coord = ((torch.arange(n_blocks[1]) 
                                             - n_blocks[1] // 2)
                                            * self.object_size[1] 
                                            / n_blocks[1])

        #every matched index in PC and PR contains the x,y coordinates for
        #each tile, i.e. (x_i,y_i) = PC.flatten()[i], PR.flatten()[i]
        PC, PR = torch.meshgrid(blocks_central_col_coord,
                             blocks_central_row_coord, indexing='xy')

        #simulate non-repeated PSFs and use symmetry
        if isinstance(n_blocks, int):
            #this is for square-shaped grids
            
            #maximum number of independent PSFs needed to be computed, the 
            #rest can be retrieved through 8-fold symmetry
            max_index_needed = n_blocks // 2 + 1
            max_psf_needed = torch.sum(torch.arange(max_index_needed + 1))
    
            #parallel compute PSFs
            ij_list = []
            for i in tqdm(range(max_index_needed)):
                for j in torch.arange(i, max_index_needed):
                    #determine the (x,y) coordinate of the point source at the object
                    #plane
                    oX = blocks_central_col_coord[j]
                    oY = blocks_central_row_coord[i]
                    
                    #Simulate the corresponding PSF. Minus sign for oX,oY is to account 
                    #for image inversion.
                    psf_complex = self.create_psf(wavelength,
                                                  self.objectdistance,
                                                  psfmag = psfmag,
                                                  pos = (-oX,-oY),
                                                  normalize=False,
                                                  prop_dist=prop_dist,
                                                  shiftpsf=shiftpsf,
                                                  debarrel=debarrel,
                                                  source=source,
                                                  propagation_medium_refractive_index=propagation_medium_refractive_index
                                                  )
                    #bring back to cpu
                    psf_complex = psf_complex.to('cpu')
                    
                    #only interested in the intensity
                    psf_temp = torch.abs(psf_complex)**2
                    
                    #crops the PSF to a size of (psf_size, psf_size)
                    psf_temp = utils.crop(psf_temp, psf_size)

                    if normalize:
                        psf[i,j,:,:] = psf_temp / torch.sum(psf_temp)
                    else:
                        psf[i,j,:,:] = psf_temp
    
            #recover remaining PSFs using diagonal symmetry
            #top left quadrant
            for i in torch.arange(1, max_index_needed):
                for j in torch.arange(i):
                    psf[i,j,:,:] = torch.flipud(torch.rot90(psf[j,i,:,:]))
    
            #bottom left quadrant
            for i in torch.arange(max_index_needed, n_blocks):
                for j in torch.arange(max_index_needed):
                    psf[i,j,:,:] = torch.flipud(psf[n_blocks - i - 1,j,:,:])
    
            #right half quadrant
            for i in torch.arange(n_blocks):
                for j in torch.arange(max_index_needed, n_blocks):
                    psf[i,j,:,:] = torch.fliplr(psf[i,n_blocks - j - 1,:,:])
                    
        elif isinstance(n_blocks, tuple):
            #this is for rectangular-shaped grids
                  
            #maximum number of independent PSFs needed to be computed, the 
            #rest can be retrieved through 4-fold symmetry
            max_index_needed_row = n_blocks[0] // 2 + 1
            max_index_needed_col = n_blocks[1] // 2 + 1
            max_psf_needed = max_index_needed_row * max_index_needed_col
    
            # parallel compute PSFs
            for i in tqdm(range(max_index_needed_row)):
                for j in range(max_index_needed_col):
                    #determine the (x,y) coordinate of the point source at the object
                    #plane
                    oX = blocks_central_col_coord[j]
                    oY = blocks_central_row_coord[i]
                    
                    #Simulate the corresponding PSF. Minus sign for oX,oY is to account 
                    #for image inversion.
                    psf_complex = self.create_psf(wavelength,
                                                  self.objectdistance,
                                                  psfmag = psfmag,
                                                  pos = (-oX,-oY),
                                                  normalize=False,
                                                  prop_dist=prop_dist,
                                                  shiftpsf=shiftpsf,
                                                  debarrel=debarrel,
                                                  source=source,
                                                  propagation_medium_refractive_index=propagation_medium_refractive_index
                                                  )
                    #bring back to cpu
                    psf_complex = psf_complex.to('cpu')
                    
                    #only interested in the intensity
                    psf_temp = torch.abs(psf_complex)**2
                    
                    #crops the PSF to a size of (psf_size, psf_size)
                    psf_temp = utils.crop(psf_temp, psf_size)

                    if normalize:
                        psf[i,j,:,:] = psf_temp / torch.sum(psf_temp)
                    else:
                        psf[i,j,:,:] = psf_temp
    
            #recover remaining PSFs using rectangular symmetry    
            #bottom left quadrant
            for i in torch.arange(max_index_needed_row, n_blocks[0]):
                for j in torch.arange(max_index_needed_col):
                    psf[i,j,:,:] = torch.flipud(psf[n_blocks[0] - i - 1,j,:,:])
    
            #right half
            for i in torch.arange(n_blocks[0]):
                for j in torch.arange(max_index_needed_col, n_blocks[1]):
                    psf[i,j,:,:] = torch.fliplr(psf[i,n_blocks[1] - j - 1,:,:])


        self.psf_coord = [blocks_central_row_coord, blocks_central_col_coord]
        self.psfs = torch.reshape(psf, (-1,psf_size, psf_size))
        return self.psfs, self.psf_coord


    
    def view_psfs(self, psfs=None):
        if psfs is None:
            psfs = self.psfs
        
        #using ImageGrid
        fig = plt.figure(dpi=200)
        grid = ImageGrid(fig, 111, nrows_ncols = self.tile_shape)
        
        for ax, psf in zip(grid, psfs):
            ax.imshow(psf)
            ax.set(xticks = [], yticks = [])
        plt.show()

    
    def fullintegral(self, wavelength=None, psfmag=1, 
                                  prop_dist=None, device='cpu', source='plane',
                                  propagation_medium_refractive_index=1):
        # check if object has been created first.
        if self.object is None:
            raise ValueError('No object to blur. Create object first.')
        m,n = self.object.shape
            
        # use len's central wavelength for point sources if not specified
        if wavelength is None:
            wavelength = self.lens_wavelength

        self.device = device
        
        # square and symmetric
        if np.allclose(self.objectx, self.objecty):
            if utils.is_even(self.object.shape[0]):
                max_index_needed = self.object.shape[0] // 2
            else:
                # odd no. of rows/cols needs to compute central row/col
                max_index_needed = self.object.shape[0] // 2 + 1

        max_psf_needed = np.sum(np.arange(max_index_needed + 1))

        # indices to loop overs
        self.img = 0
        for i in tqdm(range(max_index_needed)):
            for j in np.arange(i, max_index_needed):
                oX = self.objectx[j]
                oY = self.objecty[i]

                #psfs. Minus sign for oX,oY is to account for image inversion
                complexpsf_temp = self.create_psf(wavelength,
                                                  self.objectdistance,
                                                  psfmag=psfmag,
                                                  pos=(-oX,-oY),
                                                  shiftpsf=False,
                                                  prop_dist=prop_dist,
                                                  source=source,
                                                  propagation_medium_refractive_index=propagation_medium_refractive_index)
                psf_temp = torch.abs(complexpsf_temp)**2
                psf = psf_temp / torch.sum(psf_temp)

                #original position
                temp_img = psf * self.object[i, j]

                if utils.is_even(self.object.shape[0]):
                    #for diagonals
                    if i == j:
                        #flipud
                        temp_img += torch.flipud(psf) * self.object[m-i-1,j]
                        #fliplr
                        temp_img += torch.fliplr(psf) * self.object[i,n-j-1]
                        #flipud, fliplr
                        temp_img += torch.flipud(torch.fliplr(psf)) * self.object[m-i-1,n-j-1]
                    #for off-diagonals and non-central vertical line
                    else:
                        #flipud
                        temp_img += torch.flipud(psf) * self.object[m-i-1,j]
                        #fliplr
                        temp_img += torch.fliplr(psf) * self.object[i,n-j-1]
                        #flipud, fliplr
                        temp_img += torch.flipud(torch.fliplr(psf)) * self.object[m-i-1,n-j-1]
                        #transposed
                        temp_img += psf.T * self.object[j, i]
                        #transposed flipud
                        temp_img += torch.flipud(psf.T) * self.object[m-j-1,i]
                        #transposed fliplr
                        temp_img += torch.fliplr(psf.T) * self.object[j,n-i-1]
                        #transposed flipud, fliplr
                        temp_img += torch.flipud(torch.fliplr(psf.T)) * self.object[m-j-1,n-i-1]
                else:
                    # only perform the symmetries for non-central psf
                    if (i != max_index_needed-1) or (j != max_index_needed-1):
                        #for diagonals
                        if i == j:
                            #flipud
                            temp_img += torch.flipud(psf) * self.object[m-i-1,j]
                            #fliplr
                            temp_img += torch.fliplr(psf) * self.object[i,n-j-1]
                            #flipud, fliplr
                            temp_img += torch.flipud(torch.fliplr(psf)) * self.object[m-i-1,n-j-1]
        
                        #for central vertical line
                        elif (j == max_index_needed-1 and j != i):
                            #flipud
                            temp_img += torch.flipud(psf) * self.object[m-i-1,j]
                            #9 o'clock position
                            temp_img += torch.rot90(psf) * self.object[j, i]
                            #3 o'clock position
                            temp_img += torch.rot90(psf, k=3) * self.object[j, m-i-1]
                        #for off-diagonals and non-central vertical line
                        else:
                            #flipud
                            temp_img += torch.flipud(psf) * self.object[m-i-1,j]
                            #fliplr
                            temp_img += torch.fliplr(psf) * self.object[i,n-j-1]
                            #flipud, fliplr
                            temp_img += torch.flipud(torch.fliplr(psf)) * self.object[m-i-1,n-j-1]
                            #transposed
                            temp_img += psf.T * self.object[j, i]
                            #transposed flipud
                            temp_img += torch.flipud(psf.T) * self.object[m-j-1,i]
                            #transposed fliplr
                            temp_img += torch.fliplr(psf.T) * self.object[j,n-i-1]
                            #transposed flipud, fliplr
                            temp_img += torch.flipud(torch.fliplr(psf.T)) * self.object[m-j-1,n-i-1]
                self.img += temp_img
        return self.img
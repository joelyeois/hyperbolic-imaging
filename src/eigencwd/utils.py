"""
Utility functions.
@author: Joel Yeo, joelyeo@u.nus.edu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import next_fast_len
import numpy as np

def crop(arr, Nx, Ny=None, Nz=None):
    """
   Crops input array symmetrically.
    
    Parameters
    ----------
    arr : ndarray, 1D or 2D
        Array to be cropped
    Nx, Ny : int
        Total size of array to be cropped to become Nx x Ny. If Ny is None, then array is padded to Nx x Nx  
    
    Returns
    -------
    cropped : ndarray, 2D
        The cropped array, Nx x Ny
    ....
    """
    sh = arr.shape
    if len(sh) == 1:
        if sh[0] < Nx:
            sys.exit('N must be smaller than array dimensions.')
        cropped = arr[(sh[0] - Nx) // 2 : (sh[0] - Nx) // 2 + Nx]
    elif len(sh) == 2:
        if Ny is None:
            if sh[0] < Nx or sh[1] < Nx:
                sys.exit('N must be smaller than array dimensions.')
            cropped = arr[(sh[0] - Nx) // 2 : (sh[0] - Nx) // 2 + Nx, (sh[1] - Nx) // 2 : (sh[1] - Nx) // 2 + Nx]
        else:
            if sh[0] < Nx or sh[1] < Ny:
                sys.exit('Nx and Ny must be smaller than array dimensions.')
            cropped = arr[(sh[0] - Nx) // 2 : (sh[0] - Nx) // 2 + Nx, (sh[1] - Ny) // 2 : (sh[1] - Ny) // 2 + Ny]
    elif len(sh) == 3:
        if Ny is None and Nz is None:
            if sh[0] < Nx or sh[1] < Nx or sh[2] < Nx:
                sys.exit('N must be smaller than array dimensions.')
            cropped = arr[(sh[0] - Nx) // 2 : (sh[0] - Nx) // 2 + Nx, (sh[1] - Nx) // 2 : (sh[1] - Nx) // 2 + Nx, (sh[2] - Nx) // 2 : (sh[2] - Nx) // 2 + Nx]
        elif Ny is None or Nz is None:
            sys.exit('Please provide Nx, Ny and Nz arguments.')
        else:
            if sh[0] < Nx or sh[1] < Ny or sh[2] < Nz:
                sys.exit('Nx, Ny and Nz must be smaller than array dimensions.')
            cropped = arr[(sh[0] - Nx) // 2 : (sh[0] - Nx) // 2 + Nx, (sh[1] - Ny) // 2 : (sh[1] - Ny) // 2 + Ny, (sh[2] - Nz) // 2 : (sh[2] - Nz) // 2 + Nz]
    return cropped

def pad2d(arr, Nx, Ny = None, val=0):
    """
    Pads the input array with a constant value symmetrically.
    
    Parameters
    ----------
    arr : ndarray, 2D
        Array to be padded
    Nx, Ny : int
        Total size of array to be padded to become Nx x Ny. If Ny is None, then array is padded to Nx x Nx  
    val : float, optional
        Value of the padded element.
    
    Returns
    -------
    padded : ndarray, 2D
        The padded array, N x N
    ....
    """
    m,n = arr.shape
    if Ny is None:
        if m > Nx or n > Nx:
            sys.exit('N must be larger than array dimensions.')
        pad_length = (Nx - m) // 2
        padded = np.pad(arr, ((pad_length, pad_length), (pad_length, pad_length)), 
                        constant_values=(val,val))
    else:
        if m > Nx or n > Ny:
            sys.exit('Nx and Ny must be larger than array dimensions.')
        pad_length_x = (Nx - m) // 2
        pad_length_y = (Ny - n) // 2
        padded = np.pad(arr, ((pad_length_x, pad_length_x), (pad_length_y, pad_length_y)), 
                        constant_values=(val,val))
    return padded
    
def circle2d(N, d):
    """
    Generates 2D array for a filled-in circle.

    Parameters
    ----------
    N : int
        Length of grid in pixels
    d : int
        Diameter in pixels

    Returns
    -------
    circle : ndarray, 2D
        2D array with centered circle
    ....
    """
    circle = np.zeros((N, N))
    x = np.linspace(-1, 1, N)
    y = x
    [X, Y] = np.meshgrid(x, y)
    circle[X**2 + Y**2 <= (d / N)**2] = 1
    return circle
    

def is_even(num):
    return num % 2 == 0


def torch_shift(input, shift, padding_mode='reflection'):
    '''
    Performs subpixel shift using pytorch interpolation. Shift convention
    matches scipy.ndimage.shift

    Parameters
    ----------
    input : 2D or 3D array
        The image or volume to be shifted. Can be either real or complex.
    shift : 1D array
        The amount of shift along each axis in pixels.
    padding_mode : str
        The padding mode for outside grid values: 'zeros' | 'border' | 'reflection'.

    Returns
    -------
    sampled : 2D or 3D array
        The shifted image or volume.
    '''
    with torch.no_grad():
    
        device = input.device
        dtype = input.dtype

        if len(input.shape) == 2:
            img = input[None,...]
            h,w = input.shape
    
            dx, dy = shift
            
            x_s, y_s = (torch.arange(h, device=device)-dy)/(h-1), (torch.arange(w, device=device)-dx)/(w-1)
            x_s = torch.real(x_s.type(dtype))
            y_s = torch.real(y_s.type(dtype))
            grid_shifted = (torch.stack(torch.meshgrid(x_s, y_s, indexing='xy'), dim=-1)*2-1)
            
            if not torch.is_complex(input):
                sampled = F.grid_sample(img[None], grid_shifted[None], padding_mode=padding_mode)
            else:
                real_sampled = F.grid_sample(img[None].real, grid_shifted[None], padding_mode=padding_mode, align_corners=False)
                imag_sampled = F.grid_sample(img[None].imag, grid_shifted[None], padding_mode=padding_mode, align_corners=False)
                sampled = real_sampled + 1j * imag_sampled
        
        elif len(input.shape) == 3:
            
            vol = input[None,...]
            h,w,d = input.shape
    
            dx, dy, dz = shift

            normalized_shifts = -torch.tensor([dz,dy,dx], device=device) / torch.tensor([h,w,d], device=device) * 2
            theta = torch.hstack((torch.eye(3, device=device), normalized_shifts.reshape(3,1)))
            grid_shifted = F.affine_grid(theta[None], (1,1,h,w,d), align_corners=False)
            
            if not torch.is_complex(input):
                sampled = F.grid_sample(vol[None], grid_shifted, padding_mode=padding_mode, align_corners=False)
            else:
                real_sampled = F.grid_sample(vol[None].real, grid_shifted, padding_mode=padding_mode, align_corners=False)
                imag_sampled = F.grid_sample(vol[None].imag, grid_shifted, padding_mode=padding_mode, align_corners=False)
                sampled = real_sampled + 1j * imag_sampled
        
        return torch.squeeze(sampled)

def normalize_coordinates(xo, yo, xi, yi):
    """
    Normalizes coordinates to match Torch grid_sample input.
    
    Parameters
    ----------
    xo : 1D tensor
        x-coordinates of object to be interpolated.
    yo : 1D tensor
        y-coordinates of object to be interpolated.
    xi : 1D tensor
        x-coordinates of query points.
    yi : 1D tensor
        y-coordinates of query points.
        
    Returns
    -------
    xit, yit : tuple of 1D tensors
        Normalized x,y-coordinates of query points for Torch grid_sample.
    """
    xmin = xo[0]
    xmax = xo[-1]
    ymin = yo[0]
    ymax = yo[-1]
    xrange = xmax - xmin
    yrange = ymax - ymin
    xit = 2 * (xi - xmin) / xrange - 1
    yit = 2 * (yi - ymin) / yrange - 1
    return xit, yit

def torch_interpolate2D(img, xq, yq, mode='bilinear', padding_mode='border',
                       align_corners=False):
    """
    Performs 2D interpolation using Torch grid_sample.
    
    Parameters
    ----------
    img : 2D tensor
        2D Image to be interpolated
    xq : 1D tensor
        x-coordinates of query points.
    yq : 1D tensor
        y-coordinates of query points.
        
    Returns
    -------
    interp_img : 2D tensor
        Interpolated 2D image
    """
    interp_grid = torch.stack(torch.meshgrid(xq ,yq, indexing='xy'),
                              dim=-1)

    interp_img = F.grid_sample(img[None, None,...], interp_grid[None],
                            padding_mode=padding_mode,
                            align_corners=align_corners, mode=mode)

    return torch.squeeze(interp_img)

def fftconvolve(in1, in2, mode="full"):
    """ From scipy fftconvolve.
    
    Convolve two N-dimensional arrays using FFT.

    Convolve `in1` and `in2` using the fast Fourier transform method, with
    the output size determined by the `mode` argument.

    This is generally much faster than `convolve` for large arrays (n > ~500),
    but can be slower when only a few output values are needed, and can only
    output float arrays (int or object array inputs will be cast to float).

    As of v0.19, `convolve` automatically chooses this method or the direct
    method based on an estimation of which is faster.

    Parameters
    ----------
    in1 : torch.tensor
        First input.
    in2 : torch.tensor
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    axes : int or array_like of ints or None, optional
        Axes over which to compute the convolution.
        The default is over all axes.

    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    """

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.numel() == 0 or in2.numel() == 0:  # empty arrays
        return torch.tensor([])

    s1 = in1.shape
    s2 = in2.shape
    axes = [i for i in range(len(in1.shape))] #assume ndim convolution.

    shape = [max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
             for i in range(in1.ndim)]

    ret = _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=True)

    return _apply_conv_mode(ret, s1, s2, mode, axes)

def _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=False):
    """ From scipy.signal._signaltools
    
    Convolve two arrays in the frequency domain.

    This function implements only base the FFT-related operations.
    Specifically, it converts the signals to the frequency domain, multiplies
    them, then converts them back to the time domain.  Calculations of axes,
    shapes, convolution mode, etc. are implemented in higher level-functions,
    such as `fftconvolve` and `oaconvolve`.  Those functions should be used
    instead of this one.

    Parameters
    ----------
    in1 : torch.tensor
        First input.
    in2 : torch.tensor
        Second input. Should have the same number of dimensions as `in1`.
    axes : array_like of ints
        Axes over which to compute the FFTs.
    shape : array_like of ints
        The sizes of the FFTs.
    calc_fast_len : bool, optional
        If `True`, set each value of `shape` to the next fast FFT length.
        Default is `False`, use `axes` as-is.

    Returns
    -------
    out : torch.tensor
        An N-dimensional array containing the discrete linear convolution of
        `in1` with `in2`.

    """
    if not len(axes):
        return in1 * in2

    complex_result = (torch.is_complex(in1) or torch.is_complex(in2))

    if calc_fast_len:
        # Speed up FFT by padding to optimal size.
        fshape = [next_fast_len(shape[a], not complex_result) for a in axes]
    else:
        fshape = shape

    if not complex_result:
        fft, ifft = torch.fft.rfftn, torch.fft.irfftn
    else:
        fft, ifft = torch.fft.fftn, torch.fft.ifftn

    sp1 = fft(in1, fshape, dim=axes)
    sp2 = fft(in2, fshape, dim=axes)

    ret = ifft(sp1 * sp2, fshape, dim=axes)

    if calc_fast_len:
        fslice = tuple([slice(sz) for sz in shape])
        ret = ret[fslice]

    return ret

def _centered(arr, newshape):
    """From scipy.signal._signaltools"""
    # Return the center newshape portion of the array.
    newshape = torch.as_tensor(newshape)
    currshape = torch.tensor(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def _apply_conv_mode(ret, s1, s2, mode, axes):
    """ From scipy.signal._signaltools
    
    Calculate the convolution result shape based on the `mode` argument.

    Returns the result sliced to the correct size for the given mode.

    Parameters
    ----------
    ret : torch.tensor
        The result array, with the appropriate shape for the 'full' mode.
    s1 : list of int
        The shape of the first input.
    s2 : list of int
        The shape of the second input.
    mode : str {'full', 'valid', 'same'}
        A string indicating the size of the output.
        See the documentation `fftconvolve` for more information.
    axes : list of ints
        Axes over which to compute the convolution.

    Returns
    -------
    ret : array
        A copy of `res`, sliced to the correct size for the given `mode`.

    """
    if mode == "full":
        return ret.clone()
    elif mode == "same":
        return _centered(ret, s1).clone()
    elif mode == "valid":
        shape_valid = [ret.shape[a] if a not in axes else s1[a] - s2[a] + 1
                       for a in range(ret.ndim)]
        return _centered(ret, shape_valid).clone()
    else:
        raise ValueError("acceptable mode flags are 'valid',"
                         " 'same', or 'full'")
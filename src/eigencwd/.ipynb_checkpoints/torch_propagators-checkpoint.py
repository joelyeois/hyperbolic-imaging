import torch
from . import utils

fft2 = lambda array: torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(array)))
ifft2 = lambda array: torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(array)))

torch.set_default_dtype(torch.float64)

def scaledblas(obj, z, wavelength, dx, dxprime, dy=None, dyprime=None, 
               mprime=None, nprime=None):
    """
    Scaled band-limited angular spectrum method. 
    *Need to implement variable mprime/nprimes
    (Yu, 2012, "Band-limited angular spectrum numerical propagation method with selective scaling of observationwindow size and sample number")
    
    Parameters
    ----------
    obj : ndarray, 2D
        Array to be angular spectrum propagated.
    z : float
        Distance to propagate to [m].
    wavelength : float
        Wavelength of light [m].
    dx : float
        Sampling interval of source plane in real space, x-coordinate [m].
    dxprime : float
        Sampling interval of target plane in real space, x-coordinate [m].
    dy : float, optional
        Sampling interval of source plane in real space, y-coordinate [m].
    dyprime : float, optional
        Sampling interval of target plane in real space, y-coordinate [m].
    
    Returns
    -------
    img : ndarray, 2D
        2D diffraction pattern of array
    ....
    """
    with torch.no_grad():
        #wavenumber
        device = obj.device
        m, n = obj.shape
        if dy is None:
            dy = dx
            dyprime = dxprime
        if mprime is None and nprime is None:
            mprime = m
            nprime = n
        if mprime is not None and nprime is None:
            nprime = mprime
            
        # target coordinates
        Mprime = torch.arange(mprime, device=device) - mprime // 2
        Nprime = torch.arange(nprime, device=device) - nprime // 2
        xprime = Mprime * dxprime
        yprime = Nprime * dyprime 
        Xprime, Yprime = torch.meshgrid(xprime, yprime, indexing='xy')
        
        #frequency grid at source plane
        Lx = m * dx
        Ly = n * dy
        du = 1 / Lx
        dv = 1 / Ly
        
        #source coordinates
        u = torch.fft.fftshift(torch.fft.fftfreq(m, dx, device=device))
        v = torch.fft.fftshift(torch.fft.fftfreq(n, dy, device=device))
        u = Mprime * du
        v = Nprime * dv
        U, V = torch.meshgrid(u, v, indexing='xy')
        
        #scaling factors
        ax = dxprime / du
        ay = dyprime / dv
        
        #w coordinates
        wu = ax * u
        wv = ay * v
        dwu = wu[1] - wu[0]
        dwv = wv[1] - wv[0]
        Wu, Wv = torch.meshgrid(wu, wv, indexing='xy')
        
        #bandlimits
        Sx = dxprime * mprime
        Wx = dx * m
        Sy = dyprime * nprime
        Wy = dy * n
        uneed = 1 / wavelength / torch.sqrt(torch.tensor((2 * z / (Wx + Sx))**2 + 1, device=device))
        vneed = 1 / wavelength / torch.sqrt(torch.tensor((2 * z / (Wy + Sy))**2 + 1, device=device))
        umax = 1 / wavelength / torch.sqrt(torch.tensor((2 * z * du)**2 + 1, device=device))
        vmax = 1 / wavelength / torch.sqrt(torch.tensor((2 * z * dv)**2 + 1, device=device))
        ulim = min(umax, uneed)
        vlim = min(vmax, vneed)
        umask = torch.zeros(u.shape, device=device)
        vmask = torch.zeros(v.shape, device=device)
        umask[torch.abs(u) < ulim] = 1
        vmask[torch.abs(v) < vlim] = 1
        UVmask = torch.outer(umask, vmask)

        A1 = fft2(obj) * dx * dy
        
        #Eq. 1
        A = A1 * torch.exp(1j * 2 * torch.pi * z * torch.sqrt(0j + 1 / wavelength**2 - U**2 - V**2)) * UVmask
        #phase correction term, Shimobaba, Eq. 6, 2012. “Scaled Angular Spectrum Method.”.
        if dxprime > dx:
            phi_correction = (m * dx - mprime * dxprime) / 2
        else:
            phi_correction = - (m * dx - mprime * dxprime) / 2
        # phi_correction = 0
        A = A * torch.exp(torch.tensor(2j * torch.pi * phi_correction, device=device))
        #Eq. 9
        B = 1 / ax / ay * A * torch.exp(1j * torch.pi / ax * (ax * U)**2) * torch.exp(1j * torch.pi / ay * (ay * V)**2)
        f = torch.exp(-1j * torch.pi / ax * Wu**2) * torch.exp(-1j * torch.pi / ay * Wv**2)
        
        convolution = utils.fftconvolve(B, f, mode='same')
        
        img = torch.exp(1j * torch.pi / ax * Xprime**2) * torch.exp(1j * torch.pi / ay * Yprime**2) * convolution
        return img, xprime, yprime
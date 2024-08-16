"""
Eigenvalue column-wise decomposition (eigenCWD) algorithm.
@author: Joel Yeo, joelyeo@u.nus.edu
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

class EigenCWD:
    '''
    A class for EigenCWD.
    '''
    def __init__(self, img, obj=None, device='cpu'):
        """
        Parameters
        ----------
        img : 2D tensor
            The image to be deblurred. Should already be zero-padded
            appropriately.
        obj : 2D tensor, optional
            Groundtruth. Used for error calculation. Default: None
        device : str
            Device to compute on. Default: 'cpu';  Option: 'cuda'.
        """
        self.device = device
        self.img = img.to(self.device)
        self.obj = obj.to(self.device)
        if self.obj is None:
            self.sz = self.img.shape
        else:
            self.sz = self.obj.shape
            self.obj_norm = torch.linalg.norm(self.obj)

    def eigencwd(self, eigenpsfs, eigencoeffs, niter=100, alpha=1, mu=1e5,
                 n_components=torch.inf, verbose=True, init=None):
        """
        Optimized CWD:
            Converted fft2 -> rfft2 (33% time save)

        Parameters
        ----------
        eigenpsfs : 3D tensor
            The varying PSFs, structure (n, psf.shape[0], psf.shape[1]).
        eigencoeffs : 3D tensor
            The eigencoefficients to the eigenPSFs.
            Structure: (n, obj.shape[0], obj.shape[1]).
        niter : int, optional
            Number of iterations
        alpha, mu : float
            Method parameters. Reduce mu -> less ringing more blur.
        n_components : int
            Number of eigenPSFs to use in computation.
        verbose : boolean
            If True, prints iteration numbers
        init : 2D ndarray
            The initial guess for self.current_estimate
        """
        #register attributes
        self.n_components = n_components
        self.niter = niter
        self.alpha = torch.tensor(alpha, device=self.device)
        self.mu = torch.tensor(mu, device=self.device)
        self.eigenpsfs = eigenpsfs.to(self.device)
        self.eigencoeffs = eigencoeffs.to(self.device)
        
        #setup sizes and images
        g = self.img
        number_of_psfs = len(eigenpsfs)
        sz = self.sz
        
        #torch ffts
        rfft2 = lambda array: torch.fft.rfft2(array, dim=(0,1))
        irfft2 = lambda array: torch.fft.irfft2(array, dim=(0,1), s=self.sz)
        
        #correct array size for real FFT output
        psf_window_row_col = eigenpsfs[0].shape

        #method params
        beta = torch.sqrt(self.mu * self.alpha)
        gamma = beta

        #create matrices
        #keep the first k eigenPSFs
        k = min(n_components, number_of_psfs)

        #truncated eigenPSFs
        U = self.eigenpsfs[:k]

        #flattened, truncated eigenPSFs, shape: (P^2, k), assuming P = psf[0].numel
        h = U.reshape(k,-1).T

        #Reorganize U axes to (psf[0], psf[1], n_psfs)
        U = torch.moveaxis(U, 0, -1)

        #pseudo-inverse of h, shape: (k, P^2)
        Mplus = torch.linalg.pinv(h)

        #flattened, truncated eigencoefficients, shape: (k, M^2), where M > P.
        eigenSV = self.eigencoeffs[:k].reshape(k,-1)

        #matrix multiplication in this order prevents construction of large
        #matrix. Creates (k, k) followed by (k, M^2). Old order would create
        #(P^2, M^2), VERY BIG, followed by (k, M^2).
        # flattened_M = ((Mplus @ h) @ eigenSV).T
        flattened_M = eigenSV.T.clone()

        #initialize U
        #pad eigenPSF to image size
        padded_U = F.pad(U, (0, 0,
                             0, sz[0]-psf_window_row_col[0],
                             0, sz[1]-psf_window_row_col[1])
                         )
        shifted_padded_U = torch.roll(
            torch.roll(padded_U, -(psf_window_row_col[0] - 1) // 2,
                       dims=0), -(psf_window_row_col[0] - 1) // 2, dims=1)

        # FFT of PSFs
        fft_U = rfft2(shifted_padded_U)

        # sum_i |fft_U_i|^2
        summed_square_fft_eigenPSF = torch.sum(torch.abs(fft_U)**2, axis=2) + gamma/mu

        rep_g = rfft2(g)

        # Expand dimension to enable torch broadcasting
        rep_g = rep_g[:, :, None]
        convolved_g = irfft2(rep_g * torch.conj(fft_U))
        BTg = mu / gamma * convolved_g.reshape(-1, k)

        #create result
        flattened_MtM = gamma / beta * torch.sum(flattened_M**2, 1)

        #gradient operator, C
        C = torch.tensor([[0, -1, 0], 
                          [-1, 4 + gamma / beta, -1], 
                          [0, -1, 0]], 
                         device=self.device)

        #pad C to image size
        padC = F.pad(C, ((0, sz[0] - 3, 0, sz[1] - 3)))

        #shift gradient operator center to the left corner
        shiftpadC = torch.roll(torch.roll(padC, -1, dims=0), -1, dims=1)
        
        #Fourier domain
        FICtC = rfft2(shiftpadC)

        #create matrices
        flattened_q = torch.zeros_like(BTg)
        
        #initialization
        if init is None:
            u = torch.zeros_like(g)
        else:
            u = init
            
        self.p = torch.zeros(sz + (2,), device=self.device)
        self.v = self.p
        self.Cu = torch.zeros_like(self.p, device=self.device)
        self.mse = []
        self.relerr = []
        self.current_estimate = u

        # iterations
        for i in tqdm(torch.arange(niter)):
            if verbose:
                print('Iteration ' + str(i))
            # iterX_CWD
            # solve for x. Reshape u first to tile properly
            reshaped_u = u.ravel().reshape(-1, 1)
            flattened_Mu = flattened_M * reshaped_u

            RHS_of_2 = BTg + flattened_Mu - flattened_q
            #reshape to image size
            reshaped_RHS_of_2 = torch.reshape(RHS_of_2, sz + (k,))

            #Fourier domain
            fft_reshaped_RHS_of_2 = rfft2(reshaped_RHS_of_2)

            x = torch.sum(fft_reshaped_RHS_of_2 * fft_U, 2)
            x_norm = x / summed_square_fft_eigenPSF

            #duplicate along number_of_psf dimension
            x_norm = x_norm[:, :, None]
            xU = x_norm * torch.conj(fft_U)

            w = irfft2(fft_reshaped_RHS_of_2 - xU)

            #reshape back to flattened image space
            flattened_w = w.reshape(-1, k)

            # update q, corresponds to q^{i+1} = q^i - Mu + w
            flattened_q = flattened_q - flattened_Mu + flattened_w

            # iterU_CWD
            # solve for u
            flattened_Mwq = gamma / beta * torch.sum((flattened_w + flattened_q) * flattened_M, 1)
            vp = self.v + self.p

            #some difference operation
            Vx = F.pad(vp[:, :-1, 0], ((1, 1, 0, 0)))
            Vx = Vx[:, 1:] - Vx[:, :-1]
            Vy = F.pad(vp[:-1, :, 1], ((0, 0, 1, 1)))
            Vy = Vy[1:, :] - Vy[:-1,:]

            #flatten matrix
            flattened_RHS_of_1 = flattened_Mwq + (Vx + Vy).ravel()
            #solves (1) in Fourier domain and Fourier transform back
            fft_flattened_RHS_of_1 = rfft2(flattened_RHS_of_1.reshape(sz))
            u = irfft2(fft_flattened_RHS_of_1 / FICtC)

            # solve for v
            # soft thresholding of Cu-b -> v
            self.Cu[:,:,0] = torch.hstack((u[:, :-1] - u[:, 1:], 
                                           torch.zeros((sz[0], 1), 
                                                       device=self.device)))
            self.Cu[:,:,1] = torch.vstack((u[:-1, :] - u[1:, :], 
                                           torch.zeros((1, sz[1]),
                                                       device=self.device)))
            Dub = self.Cu - self.p
            Dubn = torch.linalg.norm(Dub, axis = 2)

            #utilize np broadcasting to recover correct matrix shape
            Dubn = Dubn[:, :, None] * torch.ones_like(Dub)

            #soft thresholding
            ratio = alpha / beta
            self.v = torch.zeros_like(Dub)
            self.v[Dubn > ratio] = (Dub[Dubn > ratio]
                               * (1 - ratio / Dubn[Dubn > ratio]))
            # update b, corresponds to p^{i+1} = p^i - Cu + v
            self.p = self.p - self.Cu + self.v
            # error metrics
            if self.obj is not None:
                # absolute errror
                mse = torch.linalg.norm(u - self.obj) / self.obj_norm
                self.mse.append(mse)
            # relative change
            relerr = (torch.linalg.norm(u - self.current_estimate)
                      / torch.linalg.norm(u))
            self.relerr.append(relerr)
            #update estimate
            self.current_estimate = u
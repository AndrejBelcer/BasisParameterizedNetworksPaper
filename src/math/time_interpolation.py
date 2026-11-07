import torch

def GetLegendreCoefficients(tau: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Evaluate `degree` shifted Legendre basis functions at a shared batch time.
    Distinct times are evaluated separately by the training wrapper's vmap.
    Returns a tensor of shape [degree], starting with the constant basis function.
    """
    # The whole image batch shares one time value.
    if tau.numel() > 1:
        # A repeated batch-time tensor is represented by its first value.
        tau = tau[0].view(1) 
    else:
        tau = tau.view(1)

    t = 2.0 * tau - 1.0
    
    polys = [torch.ones_like(t)]
    if degree > 1:
        polys.append(t)
        
    for i in range(2, degree):
        k = float(i)
        term1 = (2.0 * k - 1.0) * t * polys[-1]
        term2 = (k - 1.0) * polys[-2]
        polys.append((term1 - term2) / k)
        
    # Stack and flatten to ensure it is strictly 1D: [degree]
    return torch.stack(polys, dim=0).view(-1)


def InterpolateWeights(params: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Evaluate an effective parameter tensor from learned basis coefficients.
    
    Args:
        params (torch.Tensor): Learned coefficient tensor of shape [degree, ...].
        L (torch.Tensor): Basis-function values of shape [degree].
        
    Returns:
        torch.Tensor: Effective parameter tensor of shape [...].
    """
    return torch.einsum('i, i... -> ...', L, params)

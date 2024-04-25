import torch
def trajectory_balance_loss( 
        logZ: torch.nn.parameter.Parameter, 
        log_P_F: torch.Tensor, 
        log_P_B: torch.Tensor, 
        log_reward: torch.Tensor
) -> torch.Tensor:
    """
    Trajectory balance objective converted into mean squared error loss.
    """
    return (logZ + log_P_F - log_reward - log_P_B).pow(2)
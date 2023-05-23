'''
Optimization utilities for clip-graph
'''

def warmup_lambda(current_step: int, num_warmup_steps: int,
                  num_training_steps: int) -> float:
    '''
    Calculate the scaling factor for a linear warmup-and-decay learning rate
    schedule, as used in training BERT.

    Parameters
    ----------
    current_step: int
        The current step number.

    num_warmup_steps: int
        How many warmup steps to use.

    num_training_steps: int
        How many training steps to use.

    Returns
    -------
    The multiplicative scaling factor to apply to the learning rate.
    '''

    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    return max(
        0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
    )

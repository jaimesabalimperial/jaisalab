from datetime import datetime 


from dowel import tabular
import numpy as np

from garage import StepType
from garage.np import discount_cumsum

def get_time_stamp_as_string():
    """Get the current time stamp as a string.
    
    Returns:
        date_time_str (str) : current timestemp
    """
    # Return current timestamp in a saving friendly format.
    date_time = datetime.now()
    date_time_str = date_time.strftime("%d-%b-%Y (%H-%M-%S)")
    return date_time_str

def log_performance(itr, batch, discount, safety_discount,
                    prefix='Evaluation', is_saute=False):
    """Evaluate the performance of an algorithm on a batch of episodes.

    Args:
        itr (int): Iteration number.
        batch (SafeEpisodeBatch): The episodes to evaluate with.
        discount (float): Discount value, from algorithm's property.
        prefix (str): Prefix to add to all logged keys.

    Returns:
        numpy.ndarray: Undiscounted returns.

    """
    returns = []
    undiscounted_returns = []
    safety_returns = []
    undiscounted_safety_returns = []
    termination = []
    success = []
    for eps in batch.split():
        #account for saute algorithms
        if is_saute: 
            rewards = eps.env_infos['true_reward']
        else: 
            rewards = eps.rewards

        returns.append(discount_cumsum(rewards, discount))
        undiscounted_returns.append(sum(rewards))
        safety_returns.append(discount_cumsum(eps.safety_rewards, safety_discount))
        undiscounted_safety_returns.append(sum(eps.safety_rewards))
        termination.append(
            float(
                any(step_type == StepType.TERMINAL
                    for step_type in eps.step_types)))
        if 'success' in eps.env_infos:
            success.append(float(eps.env_infos['success'].any()))

    average_discounted_return = np.mean([rtn[0] for rtn in returns])
    average_discounted_safety_return = np.mean([rtn[0] for rtn in safety_returns])

    with tabular.prefix(prefix + '/'):
        tabular.record('Iteration', itr)
        tabular.record('NumEpisodes', len(returns))
        tabular.record('AverageDiscountedReturn', average_discounted_return)
        tabular.record('AverageReturn', np.mean(undiscounted_returns))
        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))
        tabular.record('AverageDiscountedSafetyReturn', average_discounted_safety_return)
        tabular.record('AverageSafetyReturn', np.mean(undiscounted_safety_returns))
        tabular.record('StdSafetyReturn', np.std(undiscounted_safety_returns))
        tabular.record('TerminationRate', np.mean(termination))
        if success:
            tabular.record('SuccessRate', np.mean(success))

    return undiscounted_returns
import numpy as np


def paired_permutation_test(original, mutant, num_permutations=10000, alternative='less', seed=None):
    """
    Perform a one-sided paired permutation test.

    Args:
        original (list or np.ndarray): Original values.
        mutant (list or np.ndarray): Mutant or perturbed values.
        num_permutations (int): Number of permutations to run.
        alternative (str): 'greater' or 'less'. Determines the alternative hypothesis:
            - 'greater': test if mutant > original
            - 'less': test if mutant < original
        seed (int or None): Random seed for reproducibility.

    Returns:
        p_value (float): One-sided p-value.
    """
    original = np.array(original)
    mutant = np.array(mutant)
    assert original.shape == mutant.shape, "Input lists must be of same length."

    rng = np.random.default_rng(seed)
    diffs = mutant - original
    observed_mean = np.mean(diffs)

    permuted_means = []
    for _ in range(num_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        permuted_diff = diffs * signs
        permuted_means.append(np.mean(permuted_diff))

    permuted_means = np.array(permuted_means)

    if alternative == 'greater':
        p_value = np.mean(permuted_means >= observed_mean)
    elif alternative == 'less':
        p_value = np.mean(permuted_means <= observed_mean)
    else:
        raise ValueError("alternative must be 'greater' or 'less'.")

    return p_value

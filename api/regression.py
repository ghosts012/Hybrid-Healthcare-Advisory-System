import numpy as np
def calculate_severity(logits):
    # The 'Regression' logic
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    return float(probs[0][1] * 100)
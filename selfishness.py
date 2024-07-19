from utils import *

def selfish_training(model, global_model=None, previous_model=None, previous_global_model=None, clients=10, selfishness=1):
    m = get_ravel_weights(model)
    g = get_ravel_weights(global_model)
    # Computes legitimate update
    legitimate_delta = m-g
    if previous_model is None or previous_global_model is None or global_model is None:
        # Boost the legitimate update
        selfish_delta = legitimate_delta*clients*selfishness
    else:
        # Estimate the future update by the other clients and try to counteract that
        mp = get_ravel_weights(previous_model)
        gp = get_ravel_weights(previous_global_model)
        # Computes the previous contribution to the global model
        previous_selfish_delta = mp-gp
        # Computes the overall global model update
        global_delta = g-gp
        # Computes the update from the other clients ( assumed constant between rounds)
        other_delta = global_delta*(clients) - previous_selfish_delta
        # Computes the selfish update
        selfish_delta = (legitimate_delta*clients - other_delta)*selfishness + (1-selfishness)*other_delta/(clients-1)
    model_set_weights(model, g+selfish_delta)

def downscale_aggregation(models, global_model):
    m = [ get_ravel_weights(model) for model in models ]
    g = get_ravel_weights(global_model)
    deltas = [ m_ - g for m_ in m ]
    norms = [ np.linalg.norm(delta) for delta in deltas ]
    median_norm = np.percentile(norms, [50])
    for i in range(len(deltas)):
        # If the update is too large, downscale it
        if norms[i]  > median_norm:
            m[i] = g + deltas[i]*median_norm/norms[i]
    # average all the models
    return np.mean(m, axis=0)

def rotation_aggregation(models, global_model, prev_global_model):
    m = [ get_ravel_weights(model) for model in models ]
    g = get_ravel_weights(global_model)
    deltas = [ m_ - g for m_ in m ]
    norms = [ np.linalg.norm(delta) for delta in deltas ]
    median_norm = np.percentile(norms, [50])
    marginal_median_delta = np.median(deltas, axis=0)
    for i in range(len(deltas)):
        # If the update is too large, rotate and scale it
        if norms[i]  > median_norm:
            lower_bound = 0
            upper_bound = 1
            for _ in range(10):
                beta = (lower_bound+upper_bound)/2
                # rotate the update towards the median
                rotated_delta = beta*deltas[i] + (1-beta)*marginal_median_delta
                if np.linalg.norm(rotated_delta) > median_norm:
                #if cosine_distance(rotated_delta, marginal_median_delta) < 0.3:
                    upper_bound = beta
                else:
                    lower_bound = beta
            #print(i, 1/beta)
            deltas[i] = rotated_delta
            m[i] = g + deltas[i]
    # average all the models
    return np.mean(m, axis=0)

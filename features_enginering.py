import numpy as np
from valutation import cross_valid_A_micro


# autoesplicative:

def normalize_transform(to_normalize, column_index, max_value, min_value):
    to_normalize[:, column_index] = (to_normalize[:, column_index] - min_value) / (max_value - min_value)
    return to_normalize


def center_transform(to_normalize, column_index, mean_value):
    to_normalize[:, column_index] -= mean_value
    return to_normalize


def sig_transform(to_normalize, column_index, b):
    to_normalize[:, column_index] = 1 / (1 + np.exp(b * to_normalize[:, column_index]))
    return to_normalize


def log_transform(to_normalize, column_index, b):
    to_normalize[:, column_index] = np.log(b + to_normalize[:, column_index])
    return to_normalize


def clip_transform(to_normalize, column_index, b, up_or_down="up"):
    for i in range(to_normalize.shape[0]):
        print(to_normalize[i, column_index])
        if up_or_down == "down" or up_or_down == "up_down":
            to_normalize[i, column_index] = np.sign(pow(to_normalize[i, column_index], 2)) * np.max(
                [abs(to_normalize[i, column_index]), b])
        if up_or_down == "up" or up_or_down == "up_down":
            to_normalize[i, column_index] = np.sign(pow(to_normalize[i, column_index], 2)) * np.min(
                [abs(to_normalize[i, column_index]), b])
    return to_normalize


def select_features(data, mask):
    features = range(data.shape[1])  # tutte le feature
    to_delete = np.delete(features, mask, None)  # ricavo quelle non volute
    return np.delete(data, to_delete, 1)  # elimino quelle non volute


def backward_feature_elimination(data, model, labels, min_features=1):
    n_feature = data.shape[1]
    best_features = np.array(range(n_feature))  # inizializzo a tutte le features
    best_accuracy = np.average(cross_valid_A_micro(data, labels, 5, model))
    stack = []

    while best_features.shape[0] > min_features:  # se raggiungo dimensione 1 mi fermo
        last_best = best_features
        for j in best_features:  # metto nello stack tutti i sottoinsiemi di n-1 features
            stack.append(np.delete(best_features, np.where(best_features == j), None))
        while stack:  # svuoto lo stack
            curr_features = stack.pop()
            data_filtered = select_features(data, curr_features)
            accuracy = np.average(cross_valid_A_micro(data_filtered, labels, 5, model))
            if accuracy >= best_accuracy:  # se trovo accuratezza maggiore o uguale cambio il best
                best_accuracy = accuracy
                best_features = curr_features
        if np.array_equal(last_best, best_features):
            break  # se non ho trovato una soluzione ottimizzata mi fermo
    return best_features


def forward_feature_insertion(data, model, labels, k):
    best_accuracy = 0
    best_dataset = None
    nfeat = data.shape[1]
    stop_criteria = nfeat - k
    last_best = 0

    while nfeat > stop_criteria:  # finchè il dataset non avrà k features

        if best_dataset is None:  # se il dataset è vuoto inserisco la prima feature
            for j in range(nfeat):  # cerco la feature migliore
                modified_x = data[:, j]
                accuracy = cross_valid_A_micro(modified_x, labels, 5, model)  # eseguo una validazione ad ogni passo
                if best_dataset is None or accuracy > best_accuracy:  # sostituisco la feature se ne trovo una migliore
                    best_accuracy = accuracy
                    best_dataset = modified_x
                    last_best = j

        else:  # dalla seconda iterazione in poi
            best_accuracy = 0
            temp = None  # dataset temporanio ad n+1 features
            for k in range(nfeat):  # testo una features alla volta
                modified_x = best_dataset
                modified_x = np.column_stack((modified_x, data[:, k]))
                accuracy = cross_valid_A_micro(modified_x, labels, 5, model)  # eseguo una validazione ad ogni inaerimento
                if accuracy > best_accuracy:  # se l'accuratezza è migliore, sostituisco il nuovo dataset nella lista
                    best_accuracy = accuracy
                    temp = modified_x
                    last_best = k
            best_dataset = temp

        data = np.delete(data, last_best, 1)
        nfeat = data.shape[1]
    return best_dataset

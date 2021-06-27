import numpy as np
import random as rnd


def class_matrix(labels, predictions):
    n_class = np.maximum(np.max(labels), np.max(predictions)) + 1
    ret = np.zeros((n_class, n_class))
    for i in range(labels.shape[0]):
        ret[predictions[i], labels[i]] += 1
    return ret


def confusion_matrix(labels, predictions):  # 00=TP  10=FN  01=FP  11=TN
    n_sample = labels.shape[0]
    cnf_matrix = np.zeros((2, 2))

    for i in range(n_sample):  # per ogni istanza nel test set,

        if labels[i] > 0:  # l'etichetta è 1
            if predictions[i] > 0:  # se la predizione è 1 incrementa i tp
                cnf_matrix[0, 0] += 1
            else:
                cnf_matrix[1, 0] += 1  # se la predizione è -1 incrementa i fn
        else:  # se l'etichetta è -1
            if predictions[i] < 0:
                cnf_matrix[1, 1] += 1  # se la predizione è -1 incrementa i tn
            else:
                cnf_matrix[0, 1] += 1  # se la predizione è 1 incrementa i fp
    return cnf_matrix


def confusion_matrix_multiclass(labels, predictions):  # 00=TP  10=FN  01=FP  11=TN
    n_labels = int(np.max(labels) + 1)  # numero di classi
    sample_size = labels.shape[0]  # numero istanze classification

    # inizializzo i parametri
    FP = FN = TP = TN = 0

    for i in range(n_labels):  # PER OGNI CLASSE
        for j in range(sample_size):  # calcolo i parametri

            if labels[j] == i:  # il dato appartiene alla classe che stiamo considerando
                if predictions[j] == i:
                    TP += 1  # se è predetto bene aggiorna i tp
                else:
                    FN += 1  # altrimenti aggiorna i fn
            else:  # se il dato non appartiene alla classe che stiamo considerando
                if predictions[j] != i:
                    TN += 1  # se è predetto bene aggiorna i tn
                else:
                    FP += 1  # altrimenti aggiorna i fp

    cnf_matrix = np.zeros((2, 2))  # costruisce la matrice di confusione
    cnf_matrix[0, 0] = TP
    cnf_matrix[1, 0] = FN
    cnf_matrix[0, 1] = FP
    cnf_matrix[1, 1] = TN

    return cnf_matrix


def F1_macro_average(labels, predictions):
    n_sample = labels.shape[0]
    n_class = int(np.max(labels + 1))
    P = R = 0

    for i in range(n_class):
        array_labels = -np.ones(n_sample)
        array_predictions = -np.ones(n_sample)

        # per ogni classe costruisco etichettatura e predizioni binarie
        for j in range(n_sample):
            if labels[j] == i:
                array_labels[j] = 1
            if predictions[j] == i:
                array_predictions[j] = 1

        # calcolo la matrice di confusione binaria
        matrix = confusion_matrix(array_labels, array_predictions)
        TP = matrix[0, 0]
        FP = matrix[0, 1]
        FN = matrix[1, 0]
        P += TP / (TP + FP)  # accumulo la precision
        R += TP / (TP + FN)  # accumulo la recall

    P /= n_class  # precision media
    R /= n_class  # recall media
    F1 = (2 * P * R) / (P + R)  # f1 macro average
    return F1


def A_macro_average(labels, predictions):
    n_sample = labels.shape[0]
    n_class = int(np.max(labels + 1))

    TP = FN = FP = TN = 0
    for i in range(n_class):
        array_labels = -np.ones(n_sample)
        array_predictions = -np.ones(n_sample)

        # per ogni classe costruisco etichettatura e predizioni binarie
        for j in range(n_sample):
            if labels[j] == i:
                array_labels[j] = 1
            if predictions[j] == i:
                array_predictions[j] = 1

        # calcolo la matrice di confusione binaria
        matrix = confusion_matrix(array_labels, array_predictions)
        TP += matrix[0, 0]  # accumulo tp,tn,fp,fn
        TN += matrix[1, 1]
        FP += matrix[0, 1]
        FN += matrix[1, 0]

    # calcolo tp,tn,fp ed fn medie e faccio l'accuracy
    TP /= n_class
    TN /= n_class
    FP /= n_class
    FN /= n_class

    A = (TP + TN) / (TP + TN + FP + FN)
    return A


def F1_micro_average(labels, predictions):
    matrix = confusion_matrix_multiclass(labels, predictions)
    TP = matrix[0, 0]
    FN = matrix[1, 0]
    FP = matrix[0, 1]

    P = TP / (TP + FP)  # precision
    R = TP / (TP + FN)  # recall

    F1 = (2 * P * R) / (P + R)  # f1 score
    return F1


def A_micro_average(labels, predictions):
    matrix = confusion_matrix_multiclass(labels, predictions)
    TP = matrix[0, 0]
    FN = matrix[1, 0]
    FP = matrix[0, 1]
    TN = matrix[1, 1]

    A = (TP + TN) / (TP + TN + FP + FN)
    return A


def k_split(X, k):
    sample_size = X.shape[0]
    fold_size = int(sample_size / k)

    folds_size = [fold_size] * k  # conta il numero di campioni per fold da inserire
    folds = np.array(range(k))  # contiene gli indici dei fold
    indices = np.zeros(sample_size)  # etichetta ogni campione con il numero del fold a cui apparterrà

    for i in range(sample_size):
        index = rnd.choice(folds)  # sceglie un fold casuale
        folds_size[index] -= 1  # ne diminuisce la dimensione
        if folds_size[index] == 0 and i == sample_size - 1:
            folds = folds[folds != index]
        indices[i] = index  # lo assegna all'i-esimo dato

    return indices


def cross_valid_F1_macro(Data, labels, k, model):
    sample_size = Data.shape[0]
    splits = k_split(Data, k)
    avg_f1 = 0

    for i in range(k):
        curr_train = None  # train set dell' i-esimo fold
        curr_test = None  # test set dell' i-esimo fold
        curr_train_labels = np.array([])  # etichette di train dell'iesimo-fold
        curr_test_labels = np.array([])  # etichette di test dell'iesimo fold

        for j in range(sample_size):  # per ogni campione
            if splits[j] == i:  # se fa parte del fold in considerazione lo mettiamo nel test set
                curr_test_labels = np.append(curr_test_labels, labels[j])
                if curr_test is None:
                    curr_test = Data[j]
                else:
                    curr_test = np.vstack((curr_test, Data[j]))
            else:  # altrimenti lo mettiamo nel train set
                curr_train_labels = np.append(curr_train_labels, labels[j])
                if curr_train is None:
                    curr_train = Data[j]
                else:
                    curr_train = np.vstack((curr_train, Data[j]))

        # fitta il modello e calcola le metriche
        model.fit(curr_train, curr_train_labels)
        prediction = model.predict(curr_test)
        f1 = F1_macro_average(curr_test_labels, prediction)
        avg_f1 += f1

    avg_f1 /= k
    return avg_f1


def cross_valid_A_macro(Data, labels, k, model):
    sample_size = Data.shape[0]
    splits = k_split(Data, k)
    avg_A = 0

    for i in range(k):
        curr_train = None  # train set dell' i-esimo fold
        curr_test = None  # test set dell' i-esimo fold
        curr_train_labels = np.array([])  # etichette di train dell'iesimo-fold
        curr_test_labels = np.array([])  # etichette di test dell'iesimo fold

        for j in range(sample_size):  # per ogni campione
            if splits[j] == i:  # se fa parte del fold in considerazione lo mettiamo nel test set
                curr_test_labels = np.append(curr_test_labels, labels[j])
                if curr_test is None:
                    curr_test = Data[j]
                else:
                    curr_test = np.vstack((curr_test, Data[j]))
            else:  # altrimenti lo mettiamo nel train set
                curr_train_labels = np.append(curr_train_labels, labels[j])
                if curr_train is None:
                    curr_train = Data[j]
                else:
                    curr_train = np.vstack((curr_train, Data[j]))

        # fitta il modello e calcola le metriche
        model.fit(curr_train, curr_train_labels)
        prediction = model.predict(curr_test)
        A = A_macro_average(curr_test_labels, prediction)
        avg_A += A

    avg_A /= k
    return avg_A


def cross_valid_A_micro(Data, labels, k, model):
    sample_size = Data.shape[0]
    splits = k_split(Data, k)
    avg_A = 0

    for i in range(k):
        curr_train = None  # train set dell' i-esimo fold
        curr_test = None  # test set dell' i-esimo fold
        curr_train_labels = np.array([])  # etichette di train dell'iesimo-fold
        curr_test_labels = np.array([])  # etichette di test dell'iesimo fold

        for j in range(sample_size):  # per ogni campione
            if splits[j] == i:  # se fa parte del fold in considerazione lo mettiamo nel test set
                curr_test_labels = np.append(curr_test_labels, labels[j])
                if curr_test is None:
                    curr_test = Data[j]
                else:
                    curr_test = np.vstack((curr_test, Data[j]))
            else:  # altrimenti lo mettiamo nel train set
                curr_train_labels = np.append(curr_train_labels, labels[j])
                if curr_train is None:
                    curr_train = Data[j]
                else:
                    curr_train = np.vstack((curr_train, Data[j]))

        # fitta il modello e calcola le metriche
        model.fit(curr_train, curr_train_labels)
        prediction = model.predict(curr_test)
        A = A_micro_average(curr_test_labels, prediction)
        avg_A += A

    avg_A /= k
    return avg_A


def cross_valid_F1_micro(Data, labels, k, model):
    sample_size = Data.shape[0]
    splits = k_split(Data, k)
    avg_f1 = 0

    for i in range(k):
        curr_train = None  # train set dell' i-esimo fold
        curr_test = None  # test set dell' i-esimo fold
        curr_train_labels = np.array([])  # etichette di train dell'iesimo-fold
        curr_test_labels = np.array([])  # etichette di test dell'iesimo fold

        for j in range(sample_size):  # per ogni campione
            if splits[j] == i:  # se fa parte del fold in considerazione lo mettiamo nel test set
                curr_test_labels = np.append(curr_test_labels, labels[j])
                if curr_test is None:
                    curr_test = Data[j]
                else:
                    curr_test = np.vstack((curr_test, Data[j]))
            else:  # altrimenti lo mettiamo nel train set
                curr_train_labels = np.append(curr_train_labels, labels[j])
                if curr_train is None:
                    curr_train = Data[j]
                else:
                    curr_train = np.vstack((curr_train, Data[j]))

        # fitta il modello e calcola le metriche
        model.fit(curr_train, curr_train_labels)
        prediction = model.predict(curr_test)
        f1 = F1_micro_average(curr_test_labels, prediction)
        avg_f1 += f1

    avg_f1 /= k
    return avg_f1

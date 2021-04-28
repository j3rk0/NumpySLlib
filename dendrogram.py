# %%
import numpy as np

# calcola la distanza euclidea tra cluster
def euclidean_distance(x, y):
    sum_sq = np.sum(np.square(x - y))
    return np.sqrt(sum_sq)


# calcola la massima distanza tra i punti di due cluster
def clust_distance_max(clust1, clust2):
    data1 = clust1.get_content()
    data2 = clust2.get_content()

    max_distance = None

    for point1 in data1:
        for point2 in data2:  # per ogni coppia di punti
            distance = euclidean_distance(point1, point2)
            if max_distance is None or distance > max_distance:
                max_distance = distance  # aggiorna massimo

    return max_distance


# calcola la minima distanza tra i punti di due cluster
def clust_distance_min(clust1, clust2):
    data1 = clust1.get_content()
    data2 = clust2.get_content()

    min_distance = None

    for point1 in data1:
        for point2 in data2:  # per ogni coppia di punti
            distance = euclidean_distance(point1, point2)
            if min_distance is None or distance < min_distance:
                min_distance = distance  # aggiorna minimo

    return min_distance


# calcola la distanza media tra i punti di due cluster
def clust_distance_avg(clust1, clust2):
    data1 = clust1.get_content()
    data2 = clust2.get_content()

    count = 0
    distance_sum = 0

    for point1 in data1:
        for point2 in data2:  # per ogni coppia di punti
            count += 1  # aggiorna numero di coppie
            distance_sum += euclidean_distance(point1, point2)  # aggiorna somma delle distanze

    return distance_sum / count


# %%

# struttura dati che punta a due nodi figli
class node:

    def __init__(self, left_node, right_node):
        self.lx = left_node
        self.dx = right_node
        self.level = left_node.get_level() + 1

    def get_left(self):
        return self.lx

    def get_right(self):
        return self.dx

    def get_content(self):  # ricorsivamente raccoglie i dati da tutti i figli
        return np.vstack((self.lx.get_content(), self.dx.get_content()))

    def get_level(self):
        return self.level


# ha gli stessi metodi del nodo solo che contiene i dati al posto dei figli
class leaf:
    def __init__(self, data_point):
        self.data = np.array([data_point])
        self.level = 0

    def get_content(self):
        return self.data

    def get_level(self):
        return self.level


class dendrogram:

    def __init__(self, data_points):
        self.num_leaves = data_points.shape[0]  # numero di foglie
        self.top_level = []  # stack dei cluster al livello più alto
        self.levels = 0  # livello più alto
        for i in range(self.num_leaves):  # inserisce nella lista tutte le foglie
            self.top_level.append(leaf(data_points[i]))

    def grow_a_level(self, method="max", k=None):
        stack = []

        while len(self.top_level) > 1:  # finchè non si svuota lo stack
            curr = self.top_level.pop()  # prendi un cluster
            nearest = None
            nearest_distance = None

            # trova il cluster più vicino a curr
            for cluster in self.top_level:

                # calcolo distanza tra cluster
                if method == "min":  # min linkage
                    distance = clust_distance_min(curr, cluster)
                elif method == "avg":  # average linkage
                    distance = clust_distance_avg(curr, cluster)
                else:  # max linkage
                    distance = clust_distance_max(curr, cluster)

                # aggiorna il cluster più vicino
                if nearest is None or distance < nearest_distance:
                    nearest = cluster
                    nearest_distance = distance
            # aggiunge allo stack temporaneo il nodo contenete i due cluster
            stack.append(node(curr, nearest))
            self.top_level.remove(nearest)
            # se abbiamo raggiunto k cluster fermati
            if k is not None and (len(self.top_level) + len(stack)) <= k:
                break

        # svuota stack temporaneo nello stack dei nodi top level
        while len(stack) > 0:
            self.top_level.append(stack.pop())
        self.levels += 1
        return

    def cut(self, level):  # taglia il dendrogramma ad un certo livello
        stack = []
        res = None
        i = 0
        for c_set in self.top_level:  # mette nello stack i nodi top level
            stack.append(c_set)
        while len(stack) > 0:  # finchè lo stack è pieno
            curr = stack.pop()  # prendi un elemento dallo stack
            if curr.level > level:  # se siamo troppo in alto metti i figli in stack
                stack.append(curr.get_left())
                stack.append(curr.get_right())
            else:  # se il livello è giusto
                temp = curr.get_content()  # raccogli i dati del cluster
                data_count = temp.shape[0]  # numero di dati nel cluster
                labels = np.repeat(i, data_count).reshape(data_count, 1)
                temp = np.hstack((temp, labels))  # etichetta i dati
                if res is None:
                    res = temp  # inizializza il risultato
                else:
                    res = np.vstack((res, temp))  # aggiungi al risultato
                i += 1  # aggiorna etichetta
        return res

    def grow_all(self, method="max"):  # crea il dendrogramma fino alla radice
        while len(self.top_level) > 1:
            print("grownig level: ", self.levels + 1)
            self.grow_a_level(method)

    def grow_k(self, k, method="max"):  # cresce il dendrogramma fino ad avere k cluster
        while len(self.top_level) > k:
            print("growing level: ", self.levels + 1)
            self.grow_a_level(method, k)

    def get_k_cluster(self, k):
        stack = []
        if len(self.top_level) > k:  # se l'albero è troppo basso crescilo
            self.grow_k(k)

        for c in self.top_level:  # copia il top-level in uno stack
            stack.append(c)

        if len(stack) < k:  # se l'albero è troppo cresciuto
            while len(stack) != k:  # finchè non otteniamo il numero di cluster voluto
                worst = None
                worst_distance = None
                for c in stack:  # calcolo la distanza interna di ogni elemento dello stack
                    distance = clust_distance_max(c.get_left(), c.get_right())
                    if worst is None or distance > worst_distance:
                        worst = c
                        worst_distance = distance
                stack.append(worst.get_left())  # scoppia il cluster con la maggiore distanza interna
                stack.append(worst.get_right())
                stack.remove(worst)

        res = None
        i = 0
        while len(stack) > 0:  # finchè non svuoto lo stack
            curr = stack.pop()
            temp = curr.get_content()  # prendi contenuto del nodo
            data_count = temp.shape[0]
            labels = np.repeat(i, data_count).reshape(data_count, 1)
            temp = np.hstack((temp, labels))  # etichetta i dati
            if res is None:
                res = temp  # inizializza il risultato
            else:
                res = np.vstack((res, temp))  # aggiungi al risultato
            i += 1  # aggiorna etichetta

        return res

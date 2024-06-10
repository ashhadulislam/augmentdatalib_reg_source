import numpy as np
import random
import matplotlib.pyplot as plt

class KNNOR_Reg:
    def get_label(self, X_aug, X_min, y_min, num_nbrs, alpha=0.0001):
        y_aug = []
        for i in range(X_aug.shape[0]):
            src = X_aug[i]
            distances = np.linalg.norm(X_min - src, axis=1)
            dist_indices_sorted = np.argsort(distances)
            numerator = 0
            denom = 0
            for nbr_indx in range(1, num_nbrs + 1):
                y_nbr = y_min[dist_indices_sorted[nbr_indx]]
                dist_nbr = distances[dist_indices_sorted[nbr_indx]]
                # band-aid code
                if dist_nbr == 0:
                    dist_nbr = alpha
                numerator += (1 / dist_nbr) * y_nbr
                denom += (1 / dist_nbr)

            y_src = numerator / denom
            y_aug.append(y_src)
        y_aug = np.array(y_aug)
        return y_aug

    def oversample(self, X_min, y_min, num_nbrs, proportion_of_minority, count_to_add, alpha=0.5):
        count_of_minority = int(X_min.shape[0] * proportion_of_minority)
        X_aug = []
        if count_of_minority <= num_nbrs:
            return X_aug, None
        dist_to_nth_nbr = []
        for i in range(X_min.shape[0]):
            src = X_min[i]
            distances = np.linalg.norm(X_min - src, axis=1)
            dist_indices_sorted = np.argsort(distances)
            dist_to_nth_nbr.append(distances[dist_indices_sorted[num_nbrs]])
        dist_to_nth_nbr.sort()
        threshold_dist = dist_to_nth_nbr[count_of_minority]

        while count_to_add > 0:
            for i in range(X_min.shape[0]):
                src = X_min[i]
                distances = np.linalg.norm(X_min - src, axis=1)
                sorted_indices_dist = np.argsort(distances)
                if distances[sorted_indices_dist[num_nbrs]] <= threshold_dist:
                    for j in range(1, num_nbrs + 1):
                        nbr_index = sorted_indices_dist[j]
                        X_nbr = X_min[nbr_index]
                        fractn = random.uniform(0, alpha)
                        new_point = src + fractn * (X_nbr - src)
                        src = new_point
                    X_aug.append(src)
                    count_to_add -= 1
                    if count_to_add == 0:
                        break

        X_aug = np.array(X_aug)
        y_aug = self.get_label(X_aug, X_min, y_min, num_nbrs)
        return X_aug, y_aug

    def fit_resample(self, X, y, num_nbrs=3, proportion_of_minority=0.9, bins=None, target_freq=None):
        if bins is None:
            bins = int(X.shape[0] ** 0.5)
        if not target_freq:
            hist = plt.hist(y, bins=bins)
            plt.close()
            freqs, vals = hist[0], hist[1]
            target_freq = int(np.mean(freqs))

        avg_freq = target_freq
        hist = plt.hist(y, bins=bins)
        plt.close()
        vals = hist[1]

        X_aug_all = []
        y_aug_all = []
        for i in range(len(vals) - 1):
            freq = hist[0][i]
            if freq < avg_freq:
                maj_indices = np.argwhere((y < vals[i]) | (y > vals[i + 1])).flatten()
                min_indices = np.argwhere((y >= vals[i]) & (y <= vals[i + 1])).flatten()
                X_min = X[min_indices]
                if X_min.shape[0] > num_nbrs:
                    X_maj = X[maj_indices]
                    y_maj = y[maj_indices]
                    y_min = y[min_indices]
                    count_to_add = int(abs(avg_freq - X_min.shape[0]))
                    X_aug, y_aug = self.oversample(X_min, y_min, num_nbrs, proportion_of_minority, count_to_add)
                    if len(X_aug) != 0:
                        X_aug_all.append(X_aug)
                        y_aug_all.append(y_aug)

        if len(X_aug_all) > 0:
            X_aug_all = np.concatenate(X_aug_all)
            y_aug_all = np.concatenate(y_aug_all)
            X_train_aug = np.concatenate([X_aug_all, X])
            y_train_aug = np.concatenate([y_aug_all, y])
            return X_train_aug, y_train_aug
        else:
            return X, y

# Below code is used to test the code
def main():
    import pandas as pd
    data = pd.read_csv("concrete.csv")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("X=", X.shape, "y=", y.shape)
    print("Original Regression Data shape:", X.shape, y.shape)
    plt.hist(y)
    plt.title("Original Regression Data y values")
    plt.show()
    print("************************************")
    knnor_reg = KNNOR_Reg()
    X_new, y_new = knnor_reg.fit_resample(X, y,target_freq=40)
    y_new = y_new.reshape(-1, 1)
    print("After augmentation shape", X_new.shape, y_new.shape)
    print("KNNOR Regression Data:")
    plt.hist(y_new)
    plt.title("After KNNOR Regression Data y values")
    plt.show()
    new_data = np.append(X_new, y_new, axis=1)
    print(new_data)
    print("************************************")

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np


def main():
    num = 7115
    beta = 0.8
    e = 10e-50
    M = {}
    R_new, R_old = {},{}
    # read nodes into M
    with open('dataset.txt', 'r') as data_file:
        for line in data_file:
            if line[0] == '#': continue
            key, value = line.split()
            if key not in M:
                M[key] = [value]
                R_old[key] = 1/num
                R_new[key] = (1-beta)/num
            else:
                M[key].append(value)
            if value not in M:
                M[value] = []
                R_old[value] = 1/num
                R_new[value] = (1-beta)/num
    epoch = 0
    while epoch < 500:
        for i in R_new:
            R_new[i] = (1-beta)/num
        for i in R_old:
            d_i = len(M[i])
            for dest in M[i]:
                R_new[dest] += beta*(R_old[i]/d_i)
        diff_sum = 0
        for i in R_old:
            diff_sum += np.abs(R_old[i] - R_new[i])
            R_old[i] = R_new[i]
        # print(f"R_new: ",R_new)
        # if epoch%10 == 0:
        print(f"the diff in epoch {epoch} is {diff_sum}")
        if diff_sum < e: break
        epoch += 1
    # print("the final pagerank is :")
    # print(R_new)

    with open('result.txt', 'w') as file:
        file.write(f"page\t rank\n")
        for key, values in R_new.items():
            file.write(f"{key}\t {values}\n")

if __name__ == '__main__':
    main()
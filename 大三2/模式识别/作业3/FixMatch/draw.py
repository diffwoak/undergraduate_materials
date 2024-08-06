import matplotlib.pyplot as plt
import csv

if __name__ == "__main__":
    labeled = 40
    iterations = []
    acc = []
    with open(f'result/acc_values_{labeled}.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in reader:
            iterations.append(int(row[0]))
            acc.append(float(row[1]))
    plt.clf()    
    plt.plot(iterations, acc, label='test_acc')
    plt.title(f'Labeled {labeled} Test Accuracy Over Epoches')
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.text(iterations[-1], acc[-1], f'{acc[-1]:.4f}', ha='left', va='bottom')
    plt.savefig(f'result/plt_acc_{labeled}.jpg')

import matplotlib.pyplot as plt
import numpy as np

class DGIM:
    def __init__(self, window_size):
        self.window_size = window_size
        self.buckets = []
        self.max_num = 0

    def update_buckets(self, timestamp, bit):
        if int(bit) == 1:self.buckets.append((timestamp, 1))

        while len(self.buckets) > 0 and timestamp - self.buckets[0][0] >= self.window_size:
            self.buckets.pop(0)

        i = 0
        while i < len(self.buckets) - 1:
            if self.buckets[i][1] == self.buckets[i + 1][1]:
                self.buckets[i] = (self.buckets[i][0], self.buckets[i][1] + self.buckets[i + 1][1])
                self.buckets.pop(i + 1)
            else:
                i += 1
        self.max_num = max(self.max_num, len(self.buckets))

    def count_ones(self):
        count = 0
        for _, size in self.buckets:
            count += size
        return count


# Sample usage
def main():
    window_size = 10000
    dgim = DGIM(window_size)
    timestamp = 1
    errors = []
    count_bit = 0
    record = []
    with open('TestDataStream.txt', 'r') as data_file, open('Groundtruth.txt', 'r') as groundtruth_file:
        bit = data_file.readline().strip()
        while bit and timestamp < window_size:
            # print(f"Timestamp: {timestamp}, bit: {bit}")
            dgim.update_buckets(timestamp,bit)

            if int(bit) == 1:
                count_bit += 1
                if count_bit % 12 == 0:
                    record.append(timestamp)
            
            
            timestamp += 1
            bit = data_file.readline().strip()
        truth = groundtruth_file.readline().strip()
        while bit:
            # print(f"Timestamp: {timestamp}, bit: {bit}")
            dgim.update_buckets(timestamp,bit)

            if int(bit) == 1:
                count_bit += 1
                if count_bit % (10000 % 12) == 0:
                    record.append(timestamp)
            if timestamp - record[0] > window_size:
                record.pop(0)
            count = 12* len(record)
            my_error = abs(count - int(truth))
            errors.append(my_error)

            ones_count = dgim.count_ones()
            dgim_error = abs(ones_count - int(truth))
            timestamp += 1
            bit = data_file.readline().strip()
            truth = groundtruth_file.readline().strip()
            # errors.append(dgim_error)
            if timestamp >= window_size and timestamp % 50000 == 0:
                # print(f"Timestamp: {timestamp}, DGIM Count: {ones_count}, Groundtruth: {truth}, Error: {dgim_error}")
                print(f"Timestamp: {timestamp}, DGIM Count: {count}, Groundtruth: {truth}, Error: {my_error}")
                        
        mean = np.mean(errors)
        variance = np.var(errors)
        print(f"mean: {mean}, var: {variance}")
    print(dgim.max_num)

if __name__ == '__main__':
    main()

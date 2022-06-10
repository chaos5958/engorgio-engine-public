import argparse
import os
import numpy as np

ENGORGIO_RESULT_DIR = "/workspace/research/engorgio/dataset"

def decode(log_path):
    start = []
    end = []
    with open(log_path, 'r')  as f:
        lines = f.readlines()
        num_frames = len(lines)
        for line in lines[1:]:
            result = line.split('\t')
            start.append(float(result[2]))
            end.append(float(result[3]))

    latency = end[-1] - start[0]
    throughput = num_frames / latency
    if (throughput < 59):
        print(latency, throughput, log_path)
        print('decode fail')

def infer(log_path):
    end = []
    start = []
    with open(log_path, 'r')  as f:
        lines = f.readlines()
        num_frames = len(lines)
        for line in lines[1:]:
            result = line.split('\t')
            end.append(float(result[-3]))
            start.append(float(result[2]))
            #assert(deadline > end)
    if (num_frames / (end[-1] - start[0]) < 15):
        print('infer fail')

def encode(encode_path):
    latency = []
    with open(encode_path, 'r')  as f:
        lines = f.readlines()
        for line in lines[1:]:
            result = line.split('\t')
            latency.append(float(result[4]))

    if (1 / np.average(latency) < 59):
        print('encode fail')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #dataset
    parser.add_argument('--videos', type=int, required=True)
    parser.add_argument('--log_dir', type=str, required=True)

    args = parser.parse_args()

    for i in range(args.videos):
        decode_path = os.path.join(args.log_dir, str(i), "decode_latency.txt")
        infer_path = os.path.join(args.log_dir, str(i), "infer_latency.txt")
        encode_path = os.path.join(args.log_dir, str(i), "encode_latency.txt")

        decode(decode_path)
        infer(infer_path)
        encode(encode_path)

    

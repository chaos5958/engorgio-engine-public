import sys
import os
import argparse
import glob
import json

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
    deadline = []
    with open(log_path, 'r')  as f:
        lines = f.readlines()
        num_frames = len(lines)
        for line in lines[1:]:
            result = line.split('\t')
            end.append(float(result[-3]))
            deadline.append(float(result[-2]))
            #assert(deadline > end)
    if (end[-1] >= (deadline[-1] + 1)):
        print(log_path)
        print('infer fail')

def encode(infer_path, encode_path):
    start = []
    end = []
    with open(infer_path, 'r')  as f:
        lines = f.readlines()
        num_frames = len(lines)
        for line in lines[1:]:
            result = line.split('\t')
            start.append(float(result[2]))
            end.append(float(result[-3]))

    latency = end[-1] - start[0]
    infer_throughput = num_frames / latency
    
    start = []
    end = []
    with open(encode_path, 'r')  as f:
        lines = f.readlines()
        num_frames = len(lines)
        for line in lines[1:]:
            result = line.split('\t')
            start.append(float(result[2]))
            end.append(float(result[3]))

    latency = end[-1] - start[0]
    encode_throughput = num_frames / latency
    if (encode_throughput < infer_throughput - 1):
        print('encode fail')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary_path', type=str, default='/workspace/research/engorgio-engine/build/eval/benchmark_engorgio_async')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--instance_name', type=str, default='g4dn.12xlarge')
    parser.add_argument('--num_gpus', type=int, required=True)
    parser.add_argument('--num_streams', type=int, required=True)
    parser.add_argument('--num_frames', type=int, default=None)

    args = parser.parse_args()

    if args.num_frames is None:
        cmd = f'{args.binary_path} -g {args.num_gpus} -v {args.num_streams}'
    else:
        cmd = f'{args.binary_path} -g {args.num_gpus} -v {args.num_streams} -n {args.num_frames}'
    os.system(cmd)

    log_dir = os.path.join(args.result_dir, "evaluation", args.instance_name, f's{args.num_streams}g{args.num_gpus}f7', '2022-01-15')
    for i in range(args.num_streams):
        decode_path = os.path.join(log_dir, str(i), "decode_latency.txt")
        infer_path = os.path.join(log_dir, str(i), "infer_latency.txt")
        encode_path = os.path.join(log_dir, str(i), "encode_latency.txt")
        decode(decode_path)
        infer(infer_path)
        encode(infer_path, encode_path)
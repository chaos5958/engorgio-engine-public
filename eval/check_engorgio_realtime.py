import argparse
import sys
import os

ENGORGIO_RESULT_DIR = "/workspace/research/engorgio/dataset"

def decode(log_path):
    start = []
    end = []
    assert(os.path.exists(log_path))
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
        #print(latency, throughput, log_path)
        #print('decode fail')
        return False
    return True 

def infer(log_path):
    end = []
    deadline = []
    assert(os.path.exists(log_path))
    with open(log_path, 'r')  as f:
        lines = f.readlines()
        num_frames = len(lines)
        for line in lines[1:]:
            result = line.split('\t')
            end.append(float(result[-3]))
            deadline.append(float(result[-2]))
            #assert(deadline > end)
    if (end[-1] >= (deadline[-1] + 0.1)):
        #print(log_path)
        #print('infer fail')
        return False
    return True 

def encode(infer_path, encode_path):
    start = []
    end = []
    assert(os.path.exists(infer_path))
    assert(os.path.exists(encode_path))
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
        #print('encode fail')
        return False
    return True 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #dataset
    parser.add_argument('--num_videos', type=int, required=True)
    parser.add_argument('--num_gpus', type=int, required=True)
    parser.add_argument('--log_dir', type=str, required=True)

    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir, f's{args.num_videos}g{args.num_gpus}')

    for i in range(args.num_videos):
        decode_path = os.path.join(args.log_dir, str(i), "decode_latency.txt")
        infer_path = os.path.join(args.log_dir, str(i), "infer_latency.txt")
        encode_path = os.path.join(args.log_dir, str(i), "encode_latency.txt")
        if not (decode(decode_path) and infer(infer_path) and encode(infer_path, encode_path)):
            print('[failed] real-time')
            sys.exit()
    print('[passed] real-time')

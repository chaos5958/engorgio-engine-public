
import os
perframe = {"log_dir": "/workspace/research/engorgio/result/evaluation/engorgio/Tesla_T4/s1g1_perframe/2021-12-12", "videos": 1}
engorgio1 = {"log_dir": "/workspace/research/engorgio/result/evaluation/engorgio/Tesla_T4/s16g4f5/2021-12-11", "videos": 16}
engorgio2 = {"log_dir": "/workspace/research/engorgio/result/evaluation/engorgio/Tesla_T4/s32g4f2/2021-12-11", "videos": 32}


def num_frames(log_path):
    with open(log_path, 'r')  as f:
        lines = f.readlines()
    return len(lines)

def infer_latency(log_path):
    end = []
    start = []
    with open(log_path, 'r')  as f:
        lines = f.readlines()
        for line in lines[1:]:
            result = line.split('\t')
            end.append(float(result[-3]))
            start.append(float(result[2]))
            #assert(deadline > end)
    return end[-1] - start[0]

def jpeg_latency(log_path):
    latency = []
    with open(log_path, 'r')  as f:
        lines = f.readlines()
        for line in lines[1:]:
            result = line.split('\t')
            latency.append(float(result[3])-float(result[2]))
            # print(latency)
    return sum(latency)

def libvpx_latency(log_path):
    latency = []
    with open(log_path, 'r')  as f:
        lines = f.readlines()
        for line in lines[1:]:
            result = line.split('\t')
            latency.append(float(result[4]))
    return sum(latency)

def anchor_latency(log_path):
    latency = []
    with open(log_path, 'r')  as f:
        lines = f.readlines()
        for line in lines[1:]:
            result = line.split('\t')
            latency.append(float(result[5])-float(result[4]))
            # print(latency)
    return sum(latency)


if __name__ == "__main__":
    # per-frame
    total_frames = 0
    total_encode = 0
    videos = perframe['videos']
    log_dir = perframe['log_dir']
    total_infer = infer_latency(os.path.join(log_dir, str(0), "infer_latency.txt"))
    for i in range(videos):
        total_frames += num_frames(os.path.join(log_dir, str(i), "decode_latency.txt"))
        total_encode += libvpx_latency(os.path.join(log_dir, str(i), "encode_latency.txt"))
    print(0, total_infer / total_frames / 4 * 1000, total_encode / total_frames * 1000)

    # selective
    print(0, total_infer / total_frames / 4 * 1000 / 20, total_encode / total_frames * 1000)
    print(0, total_infer / total_frames / 4 * 1000 / 40, total_encode / total_frames * 1000)

    total_frames = 0
    total_encode = 0
    videos = engorgio1['videos']
    log_dir = engorgio1['log_dir']
    total_infer = infer_latency(os.path.join(log_dir, str(0), "infer_latency.txt"))
    total_anchor = anchor_latency(os.path.join(log_dir, "anchor_latency.txt"))
    for i in range(videos):
        total_frames += num_frames(os.path.join(log_dir, str(i), "decode_latency.txt"))
        total_encode += jpeg_latency(os.path.join(log_dir, str(i), "encode_latency.txt"))
    print(total_anchor / total_frames * 1000, total_infer / total_frames * 1000, total_encode / total_frames * 1000)

    total_frames = 0
    total_encode = 0
    videos = engorgio2['videos']
    log_dir = engorgio2['log_dir']
    total_infer = infer_latency(os.path.join(log_dir, str(0), "infer_latency.txt"))
    total_anchor = anchor_latency(os.path.join(log_dir, "anchor_latency.txt"))
    for i in range(videos):
        total_frames += num_frames(os.path.join(log_dir, str(i), "decode_latency.txt"))
        total_encode += jpeg_latency(os.path.join(log_dir, str(i), "encode_latency.txt"))
    print(total_anchor / total_frames * 1000, total_infer / total_frames * 1000, total_encode / total_frames * 1000)



import os
import argparse
import glob
import json

contents = ['chat0', 'lol0', 'gta0', 'valorant0', 'minecraft0', 'fortnite0']
instances = ['xlarge', '2xlarge', '4xlarge', '12xlarge']

baseline_models = ['EDSR_B8_F16_S3', 'EDSR_B8_F20_S3', 'EDSR_B8_F24_S3', 'EDSR_B8_F28_S3', 'EDSR_B8_F32_S3']
baseline_anchors = [6, 8, 10, 12]
model = 'EDSR_B8_F32_S3'
anchors = 3
epoch_length = 40

input_resolution = 720
output_resolution = 2160
framerate = 60

def get_num_cpus(instance_name):
    if instance_name == 'xlarge':
        return 4
    elif instance_name == '2xlarge':
        return 8
    elif instance_name == '4xlarge':
        return 16
    elif instance_name == '12xlarge':
        return 48

def get_num_gpus(instance_name):
    if instance_name == 'xlarge':
        return 1
    elif instance_name == '2xlarge':
        return 1
    elif instance_name == '4xlarge':
        return 1
    elif instance_name == '12xlarge':
        return 4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--instance_name', type=str, default='g4dn')
    args = parser.parse_args()

    # Load inference, encoding throughput
    infer_throughputs = {}
    for model in baseline_models:
        json_path = os.path.join(args.result_dir, 'evaluation', args.instance_name, f'{input_resolution}p', model, 'infer_result.json')
        with open(json_path, 'r') as f:
            infer_throughputs[model] = json.load(f)

    encode_throughputs = {}
    json_path = os.path.join(args.result_dir, 'evaluation', args.instance_name, f'{output_resolution}p', 'software_encode_result.json')
    with open(json_path, 'r') as f:
        encode_throughputs['software'] = json.load(f)
    json_path = os.path.join(args.result_dir, 'evaluation', args.instance_name, f'{output_resolution}p', 'hardware_encode_result.json')
    with open(json_path, 'r') as f:
        encode_throughputs['hardware'] = json.load(f)

    # Load per-content configuration
    json_path = os.path.join(args.result_dir, 'section8', 'configuration.json')
    with open(json_path, 'r') as f:
        configurations = json.load(f)

    # Run baseline emulation per content per instance
    for instance in instances:
        results = {}
        num_cpus = get_num_cpus(instance)
        num_gpus = get_num_gpus(instance)
        # print(num_cpus, num_gpus)
        for content in contents:
            results[content] = {}
            # per-frame
            num_blocks, num_channels = configurations[content]['per-frame']['num_blocks'], configurations[content]['per-frame']['num_channels']
            model = f'EDSR_B{num_blocks}_F{num_channels}_S3'
            num_infer_streams = (infer_throughputs[model]['throughput'] * num_gpus) // framerate
            num_hw_encode_streams = (encode_throughputs['hardware']['throughput'] // framerate) * num_gpus

            num_encode_cpus = num_cpus - 1 - 3 * num_gpus
            min_threads = None
            for num_threads in encode_throughputs['software'].keys():
                if encode_throughputs['software'][num_threads]['throughput'] >= framerate:
                    min_threads = int(num_threads)
                    break
            num_sw_encode_streams = num_encode_cpus // min_threads

            results[content]['per-frame-sc'] = min(num_infer_streams, num_sw_encode_streams)
            results[content]['per-frame-hc'] = min(num_infer_streams, num_hw_encode_streams)

            # selective
            anchors = configurations[content]['selective']['avg_anchors']
            fraction = anchors / epoch_length
            num_blocks, num_channels = 8, 32
            model = f'EDSR_B{num_blocks}_F{num_channels}_S3'
            num_infer_streams = ((infer_throughputs[model]['throughput'] / fraction) * num_gpus) // framerate
            num_hw_encode_streams = (encode_throughputs['hardware']['throughput'] // framerate) * num_gpus
            # print(fraction, infer_throughputs[model]['throughput'] / fraction, num_infer_streams)

            num_encode_cpus = num_cpus - 1 - 3 * num_gpus
            min_threads = None
            for num_threads in encode_throughputs['software'].keys():
                min_threads = int(num_threads)
                if encode_throughputs['software'][num_threads]['throughput'] >= framerate:
                    break
            num_sw_encode_streams = num_encode_cpus // min_threads

            results[content]['selective-sc'] = min(num_infer_streams, num_sw_encode_streams)
            results[content]['selective-hc'] = min(num_infer_streams, num_hw_encode_streams)

        json_dir =  os.path.join(args.result_dir, 'section8', f'{args.instance_name}.{instance}')
        os.makedirs(json_dir, exist_ok=True)
        json_path = os.path.join(json_dir, 'baseline_throughput.json')
        with open(json_path, 'w') as f:
            json.dump(results, f)  

        print(results)
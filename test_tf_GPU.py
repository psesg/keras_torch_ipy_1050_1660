
import os
import subprocess as sp

# for Tensorflow Suppressing Warning messages
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed '''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

gpu = tf.config.list_physical_devices('GPU')
# gpu = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(gpu)):
    print(gpu[i])
    details = tf.config.experimental.get_device_details(gpu[i])
    device_name = details.get('device_name', 'Unknown GPU')
    print(f'device name: {device_name}')
    compute_capability = details.get('compute_capability', 'Unknown')
    print(f'compute capability: {compute_capability[0]}.{compute_capability[1]}')
    mem = get_gpu_memory()
    print(f'device memory: {mem[0]} MB')

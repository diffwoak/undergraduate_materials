import torch

def save_tensor(tensor, filename):
    tensor.numpy().astype('float32').tofile(filename)

# Define input and kernel
input_tensor = torch.randn(1, 3, 256, 256)   # (batch_size, channels, height, width)
kernel_tensor_1 = torch.randn(1, 3, 3, 3)    # (out_channels, in_channels, kernel_height, kernel_width)
kernel_tensor_2 = torch.randn(1, 3, 3, 3)    # (out_channels, in_channels, kernel_height, kernel_width)
kernel_tensor_3 = torch.randn(1, 3, 3, 3)    # (out_channels, in_channels, kernel_height, kernel_width)

# Save tensors to binary files
save_tensor(input_tensor, 'input_tensor.bin')
save_tensor(kernel_tensor_1, 'kernel_tensor_1.bin')
save_tensor(kernel_tensor_2, 'kernel_tensor_2.bin')
save_tensor(kernel_tensor_3, 'kernel_tensor_3.bin')

# Perform convolution
output_tensor_1 = torch.nn.functional.conv2d(input_tensor, kernel_tensor_1, stride=1, padding=0)
output_tensor_2 = torch.nn.functional.conv2d(input_tensor, kernel_tensor_2, stride=2, padding=1)
output_tensor_3 = torch.nn.functional.conv2d(input_tensor, kernel_tensor_3, stride=3, padding=1)

# Save tensors to binary files
save_tensor(output_tensor_1, 'output_tensor_1.bin')
save_tensor(output_tensor_2, 'output_tensor_2.bin')
save_tensor(output_tensor_3, 'output_tensor_3.bin')

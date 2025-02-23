import torch
print("CUDA Available:", torch.cuda.is_available())  # Should print True
print("GPU Name:", torch.cuda.get_device_name(0))  # Should print "GeForce GTX 1650"
print("CUDA Version:", torch.version.cuda)  # Should print "12.x"

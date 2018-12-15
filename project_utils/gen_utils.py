import torch


class GenUtils():
    @staticmethod
    def get_device():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    @staticmethod
    def set_device_config():
        if GenUtils.get_device().type == 'cuda':
            torch.device('cuda')
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

    @staticmethod
    def is_cuda():
        if GenUtils.get_device().type == 'cuda':
            return True

def get_tensor_statistics_str(tensor, name="", formatting="standard"):
    """ Returns string of formatted tensor statistics, contains min, max, mean, and std"""
    if isinstance(tensor, (torch.FloatTensor, torch.cuda.FloatTensor)):
        if formatting == "standard":
            string = "elem in [{:6.3f}, {:6.3f}]    mean: {:6.3f}    std: {:6.3f}    size: {}".format(tensor.min().item(), tensor.max().item(), tensor.mean().item(), tensor.std().item(), tuple(tensor.size()))
        elif formatting == "short":
            string = "[{:6.3f}, {:6.3f}]  mu: {:6.3f}  std: {:6.3f}  {!s:17} {: 6.1f}MB".format(tensor.min().item(), tensor.max().item(), tensor.mean().item(), tensor.std().item(), tuple(tensor.size()), 4e-6 * prod(tensor.size()))
    elif isinstance(tensor, (torch.LongTensor, torch.ByteTensor, torch.cuda.LongTensor, torch.cuda.ByteTensor)):
        tensor = tensor.to('cpu')
        string = "elem in [{:6.3f}, {:6.3f}]    size: {}    HIST BELOW:\n{}".format(tensor.min().item(), tensor.max().item(), tuple(tensor.size()), torch.stack([torch.arange(0, tensor.max()+1), tensor.view(-1).bincount()], dim=0))
    else:
        raise NotImplementedError("A type of tensor not yet supported was input. Expected torch.FloatTensor or torch.LongTensor, got: {}".format(tensor.type()))
    string = string + "    " + name
    return string

def print_tensor_statistics(tensor, name="", formatting="standard"):
    print(get_tensor_statistics_str(tensor, name, formatting))

def get_weight_statistics_str(layer, name="", formatting="standard"):
    return get_tensor_statistics_str(layer.weight, name, formatting)

def get_model_size_str(model):
    nelem = 0
    for module in model.modules():
        if hasattr(module, 'weight'):
            nelem += module.weight.numel()
        if hasattr(module, 'bias'):
            nelem += module.weight.numel()
    size_str = "{:.2f} MB".format(nelem * 4 * 1e-6)
    return size_str

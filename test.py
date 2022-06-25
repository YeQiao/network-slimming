import torch 
from scipy import stats 
from torchvision.models import alexnet, resnet18, mobilenet_v3_small
import numpy as np
import matplotlib.pyplot as plt

# model = mobilenet_v3_small(pretrained=True)
model = resnet18(pretrained=True)

def plot_hist_conv_linear(model,save_fig=False,plt_show=True,model_name=None):
    layers = {}
    weights = {}
    counter = 0
    if model_name == None:
        model_name = model.__class__.__name__
    else:
        model_name= model_name
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            layers[layer.__class__.__name__ + "_" + str(counter)] = "x".join(map(str, layer.weight.shape))
            weights[layer.__class__.__name__ + "_" + str(counter)] = layer.weight.cpu().detach().numpy().flatten()

        if isinstance(layer, torch.nn.Linear):
            layers[layer.__class__.__name__ + "_" + str(counter)] = "x".join(map(str, layer.weight.shape))
            weights[layer.__class__.__name__ + "_" + str(counter)] = layer.weight.cpu().detach().numpy().flatten()
        if isinstance(layer,torch.nn.Conv2d) or isinstance(layer,torch.nn.Conv2d):
            counter += 1
    for idx, params in weights.items():
        (mean_fitted, std_fitted) = stats.norm.fit(params)
        x = np.linspace(min(params), max(params), 600)
        weight_guass_fit = stats.norm.pdf(x, loc=mean_fitted, scale=std_fitted)
        n, bins, patchers = plt.hist(params, histtype='stepfilled',
                                     cumulative=False, bins=600, density=True)

        plt.plot(x, weight_guass_fit, label='guess')
        plt.title(idx + " : " + layers[idx])
        plt.legend()
        if save_fig == True:
            figure_name = creating_path("reports","filters","distrbutions",model_name,file_name=idx + "__" + layers[idx],extension='png')
            plt.savefig(figure_name, dpi=150, bbox_inces='tight')
        if plt_show == True:
            plt.show()

plot_hist_conv_linear(model,save_fig=False,plt_show=True,model_name=None)
# SparseLUT
Sparse Connectivity Optimization for Lookup Table-based Deep Neural Networks

Paper: https://arxiv.org/abs/2503.12829

An example of the connectivity curve is shown in this figure (epoch=200, T=160).
<img src="example.PNG" alt="Description" width="2000">


We have shared the optimized connectivity masks obtained from SparseLUT with all baseline models listed in Table IV of the paper.

In addition, we provide the ```LoadFeatureMask``` function, which can be used to seamlessly load these masks into frameworks such as **LogicNets**, **PolyLUT**, **PolyLUT-Add**, and **NeuraLUT**. By utilizing this function, you can replace the default random connectivity in these models with the optimized connectivity achieved through SparseLUT.

```
def LoadFeatureMask(out_features: int, fan_in: int, layer_number: int):
    """
    Load a pre-calculated feature mask for a specific layer from a CSV file.

    Args:
        out_features (int): Number of output features.
        fan_in (int): Number of non-zero connections per output.
        layer_number (int): Layer number (used to identify the corresponding CSV file).

    Returns:
        torch.Tensor: The feature mask as a torch tensor.
    """
    # Construct the file name based on the layer number
    file_name = f"./mask_layer_{layer_number}.csv"

    # Load the CSV file(import pandas as pd)
    df = pd.read_csv(file_name, header=None)
    df = df.iloc[1:, :]  # Ignore the first row (column numbers)

    # Validate the loaded mask dimensions
    assert df.shape == (out_features, fan_in), \
        f"Loaded mask shape {df.shape} does not match expected shape ({out_features}, {fan_in})"
    
    # Convert the DataFrame to a torch tensor and ensure it's on GPU
    feature_mask = torch.tensor(df.values, dtype=torch.long).cuda()
    
    # Ensure the mask is sorted for each row (optional, if required)
    feature_mask = torch.sort(feature_mask, dim=1).values
    
    return feature_mask
```
    
This functionality allows researchers to directly integrate SparseLUT’s optimized connectivity into their work, enabling comparisons or further enhancements on various frameworks.

The complete code for SparseLUT will be made publicly available on this repository later.

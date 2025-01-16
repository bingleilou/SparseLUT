# SparseLUT
Sparse Connectivity Optimization for Lookup Table-based Deep Neural Networks


We have shared the optimized connectivity masks obtained from SparseLUT with all baseline models listed in Table IV of our paper.

In addition, we provide the FeatureMasks_Load function, which can be used to seamlessly load these masks into frameworks such as **LogicNets**, **PolyLUT**, **PolyLUT-Add**, and **NeuraLUT**. By utilizing this function, you can replace the default random connectivity in these models with the optimized connectivity achieved through SparseLUT.

This functionality allows researchers to directly integrate SparseLUT’s optimized connectivity into their work, enabling comparisons or further enhancements on various frameworks.

The complete code for SparseLUT will be made publicly available on this repository after the paper is accepted.
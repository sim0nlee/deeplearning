# Mitigating depth-induced challenges via close-to-linear neural networks

In the context of deep neural networks, we offer an overview of two approaches that have been proposed to tackle the issue of degenerate gradients. The first one consists of a so-called ”shaping” of the ReLU activation function, while the second employs the well-known residual connections with scaled residual branches. We found that these remedies are indeed successful in making simple but very deep architectures trainable through the role of the extra parameters they introduce in the model. Also, we present the advantages of setting these parameters as trainable rather than constant.

------------------

Python version: 3.11

To install the requirements run 

`pip install -r requirements.txt`

The `main.py` file contains the script that we used to run our experiments. You can adjust the various model, training and miscellaneous hyperparameters to test different combinations.







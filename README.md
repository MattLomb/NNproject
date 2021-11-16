# NNproject
Neural network project of generation of images by text based on StyleGAN2 and CLIP

# HOW  TO RUN

Clone this repository

```
git clone https://github.com/MattLomb/NNproject.git && cd NNproject
```

Create the environment
```
python3 -m venv env
```

Activate the environment
```
source env/bin/activate
```
Install the python packages
```
pip install -r requirements.txt
```
Download the weights

```bash
cd CLIP/weights
chmod 777 download_weights.sh
./download_weights.sh

cd ../../StyleGAN2/weights/ && python3 download.py
```

You can run the software with:

```
python3 run.py --config <config> --target <target>
```

Specifying `<config>` according to the following table:

|        Config        |                                   Meaning                                  |
|:--------------------:|:--------------------------------------------------------------------------:
|   StyleGAN2_ffhq_d   |             Use StyleGAN2-ffhq to solve the Text-to-Image task             |
|  StyleGAN2_ffhq_nod  |  Use StyleGAN2-ffhq without Discriminator to solve the Text-to-Image task  |

And in `<target>` the input text for image generation
You will find the results in the folder `./tmp`, a different output folder can be specified with `--tmp-folder`

Example:

```bash
python3 run.py --config "StyleGAN2_ffhq_d" --target "A woman wearing eyeglasses with blond hair"
```

### Suggestion for developing
After the installation of a pip packages into the environment, 
it's necessary to update the requirments.txt running this command:

```
pip freeze > requirements.txt
```

# References

This repository uses these projects:
- [StyleGAN2-TensorFlow-2.X](https://github.com/rosasalberto/StyleGAN2-TensorFlow-2.x)
- [nl image search](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/nl_image_search.ipynb)

# WP-CLIP
This is the official repository of the paper 
## Leveraging CLIP to Predict Wölfflin’s Principles in Visual Art

[Abhijay Ghildyal](https://abhijay9.github.io/), [Li-Yun Wang](https://www.linkedin.com/in/li-yunwang/), and [Feng Liu](https://pages.cs.wisc.edu/~fliu/). 

In ICCV AI4VA workshop, 2025 (Oral). Please checkout the paper on [[Arxiv]](https://arxiv.org/abs/2508.12668)

### Examples of Wölfflin’s principles predicted for famous paintings using our metric, WP-CLIP
![teaser](https://github.com/abhijay9/wpclip/blob/main/samples/pandora_samples.png)

## Abstract
Wölfflin's five principles offer a structured approach to analyzing stylistic variations for formal analysis. However, no existing metric effectively predicts all five principles in visual art. Computationally evaluating the visual aspects of a painting requires a metric that can interpret key elements such as color, composition, and thematic choices. Recent advancements in vision-language models (VLMs) have demonstrated their ability to evaluate abstract image attributes, making them promising candidates for this task. In this work, we investigate whether CLIP, pre-trained on large-scale data, can understand and predict Wölfflin's principles. Our findings indicate that it does not inherently capture such nuanced stylistic elements. To address this, we fine-tune CLIP on annotated datasets of real art images to predict a score for each principle. We evaluate our model, WP-CLIP, on GAN-generated paintings and the Pandora-18K art dataset, demonstrating its ability to generalize across diverse artistic styles. Our results highlight the potential of VLMs for automated art analysis.

## Run on a single image
```
# install requirements
pip install torch
pip install torchvision
pip install openai-clip

# download model
pip install gdown
gdown 1IkAmA2pIyiMTWVgg-W1U193Zd3-MwOeQ --output ./ckpts/

# run on a sample image
python test_single_image.py -i samples/26987.jpg
```

For detailed installation instructions, check [CLIP-IQA](https://github.com/IceClear/CLIP-IQA).

The model weights can be downloaded from [this link](https://drive.google.com/drive/folders/1QpHuSoFR9EE-9XeB1W7tTNZGnrKrGOHL?usp=sharing)

## Citation

If you find this repository useful for your research, please cite the following paper.

```
@inproceedings{ghildyal2025wpclip,
  title={WP-CLIP: Leveraging CLIP to Predict Wölfflin's Principles in Visual Art},
  author={Abhijay Ghildyal and Li-Yun Wang and Feng Liu},
  booktitle={International Conference on Computer Vision (ICCV) AI for Visual Arts Workshop},
  year={2025}
}
```

![CLIP](https://raw.githubusercontent.com/joaquincabezas/clip_is_awesome/main/images/clip_is_awesome.png)
# CLIP is awesome 

## ELI5 (Explain like I'm 5 years old)
CLIP is able to connect the meaning of a text with the meaning of an image. It converts the image into a long list of numbers. It also converts the text into a long list of numbers. Now that we only have numbers (hundreds of them!), we can compare both lists and check if the meaning of the text is similar to the meaning of the image. We can ask CLIP if a dog is in the image so we don't have to check every image manually. For building CLIP, the system was teached with millions of images with its description, so it could learn from the examples.

## CLIP in a nutshell

CLIP (Contrastive Language–Image Pre-training) is a neural network which efficiently learns visual concepts from natural language supervision, through the use of Contrastive Learning. It can be used as a classification tool where we can compare the embeddings of an image 

Links:

- [CLIP: Connecting Text and Images](https://openai.com/blog/clip/)
- [CLIP paper on Arxiv: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [The Beginner’s Guide to Contrastive Learning](https://www.v7labs.com/blog/contrastive-learning-guide)

| ![CLIP](https://raw.githubusercontent.com/joaquincabezas/clip_is_awesome/main/images/CLIP.png) |
|:--:|
| Image Credit: https://github.com/openai/CLIP |

As seen in the picture above, during the pre-training phase, both encoders (image and text) are trained with image-text pairs and incentivized to create similar representations for the corresponding pairs (and not similar for the negative examples). For using CLIP as a Zero-Shot classifier, one will create a text that defines the class, and the will obtain its representation with the text encoder. Finally, the representation of the image is compared with each class with some distance metric (i.e. cosine similarity) and we can select the class that is closest.

## OpenCLIP

CLIP is a OpenAI product and it is available at [Github](https://github.com/openai/CLIP) but we will be using an alternative that's been trained on LAION (Large-scale Artificial Intelligence Open Network) and that provides multiple ViT sizes. OpenCLIP is provided by mlfoundations in the following repo: [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip).

## LAION

[LAION](https://laion.ai/) is a non-profit organization that provides datasets, tools and models to liberate machine learning research. In particular, for openCLIP, it's the dataset provider for the most downloaded model, with LAION-2B (the English subset of LAION-5B). You can find the models in [HuggingFace](https://huggingface.co/collections/laion/openclip-laion-2b-64fcade42d20ced4e9389b30).
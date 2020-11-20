# Oscar: Object-Semantics Aligned Pre-training for Image Captioning Task

## Background

**Image Captioning** generates a description of an image. Thus, it involves both computer vision(CV) and natural language processing(NLP). CV is needed to understand the image, specifically to detect and recognize features, objects, scenes, etc. NLP is needed to generate syntactically and semantically well-formed captions.

The dataset for image captioning generally consists of two inputs: image and text.

![]()

The model used for image captioning, generally consist of a CNN model(ENCODER) that would produce a hidden state h.
Then, it would decode this hidden state by using a LSTM(DECODER) and generate recursively each word of the caption. The problem with this model is that it is using the whole representation of the image denoted by h to predict the next word which would generally represent a portion of the image.

![]()

To overcome this, attention mechanism was used to focus on each portion of the image while predicting next word. With an Attention mechanism, the image is first divided into n parts, and we compute with a Convolutional Neural Network (CNN) representations of each part h1,…, hn. When the RNN is generating a new word, the attention mechanism is focusing on the relevant part of the image, so the decoder only uses specific parts of the image.


## OSCAR

It is a novel pre-training technique for Vision-Language(VL) Tasks like image captioning.

### Problems with previous implementations

1) The image features are extracted using object detectors in which the regions are over-sampled like in Faster RCNN. So, the extracted regions might contain portions of other objects implying that the image features contain a mix of features of several objects. This introduces ambiguousity in the data itself.

2) Since there are no explicit labelled alignments between objects or regions of the image and the words within the caption. It is difficult for the model to learn this alignment.

### In Simple Terms.....

While going through several datasets of VL, it was observed that the main objects of any image are usually mentioned in the text. For example, on the MS COCO dataset, the percentages that an image and its paired text share at least 1, 2, 3 objects are 49:7%, 22:2%, 12:9%, respectively. This led to an idea of using these main objects for aligning image data and text data. Since these main objects act as anchors around which rest of the objects or regions within an image align with the rest of the words in the text, these are known as **anchor points**.

These acts as labels for the alignment problems and hence the input now sent for pre-training consists of 3 parts:
    1) Region features
    2) Object Tags
    3) Text


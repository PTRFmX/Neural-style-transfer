# Neural-style-transfer  
Simple implementation of neural style transfer

## Requirements  

### Data Files  

* [Pretrained model VGG19](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)

### Dependencies  

* Tensorflow
* Pillow
* Numpy
* Scipy
* Argparse
* Python-reize-image

## Running  

`python neural_style.py --content <content_img_path> --style <style_img_path> --output <output_img_path> --model <model_path>`  

### Arguments  

*Required:*  
--content: content file path  
--style: style file path  
--output: output file path  
--model: model path (example: 'pretrained_model/imagenet-vgg-verydeep-19.mat')  
  
*Optional:*  
--iterations: number of iterations that you'd like to train  
--content-weight: the weight of content loss  
--style-weight: the weight of style loss  
--learning-rate: the learning rate of the adam optimizer  

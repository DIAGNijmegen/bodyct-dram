# Dense Regression Activation Maps For Lesion Segmentation in CT scans of COVID-19 patients


the implementation of our [dense regression activation map algorithm](https://arxiv.org/pdf/2105.11748.pdf).
The algorithm takes a CT image and its corresponding lobe segmentation as the input, generating a lesion segmentation as the output.


## Installation

 - Please check `/docker_base/DockerFile` for the required pacakges to build the docker image. 
   Note that there is another `/DockerFile` is for build a docker image for grand-challenge [algorithm](https://grand-challenge.org/algorithms/weakly-supervised-emphysema-subtyping/).
 - Regarding DGL library, we suggest you install 0.6.x. 0.4.x cannot be used because of bugs related to the implementation of graph attention networks.



## Usage

- before training, run `prepare_data.py` to generate lobw-wise chunk images for training.  

- For training, run `train.py`
- For testing, run `process_pipeline.py`

## License
[MIT](https://choosealicense.com/licenses/mit/)
import logging

import azure.functions as func
import torch
from PIL import Image
from torchvision import transforms
import io


def main(req: func.HttpRequest) -> func.HttpResponse:
    logger = logging.getLogger('visioner')
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    logger.debug('Python HTTP trigger function processed a request.')

    if not req.files:
        return func.HttpResponse(
                "Missing image field! Exiting...",
                status_code=400
            )
    
    if len(req.files.getlist('image')) > 1:
        return func.HttpResponse(
                "Please post only one file at a time!",
                status_code=400
            )


#   TODO rewrite to handle one
    for input_file in req.files.values():
        filename = input_file.filename
#        contents = input_file.stream.read()

        logger.info('Filename: %s' % filename)
#        logger.info('Contents:')
#        logger.info(contents)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    model.eval()

        # sample execution (requires torchvision)
    input_image = Image.open(input_file.stream)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
#    print(output[0])
     # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)



    return func.HttpResponse(f"This HTTP triggered function executed successfully, image field is present.")

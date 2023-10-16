import logging

import azure.functions as func
import torch


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    model.eval()

    image = req.params.get('image')
    if not image:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            image = req_body.get('image')

    if image:
        return func.HttpResponse(f"This HTTP triggered function executed successfully, image field is present.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully, but image field is not present in the request.",
             status_code=200
        )

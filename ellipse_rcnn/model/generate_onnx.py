import torch
from torch.utils.data import DataLoader
import onnx

from detection.training import CraterEllipseDataset, collate_fn
from src import CraterDetector

if __name__ == "__main__":
    model = CraterDetector()
    model.load_state_dict(torch.load("blobs/CraterRCNN.pth"))
    model.eval()
    device = torch.device('cpu')
    model.to(device)

    ds = CraterEllipseDataset(file_path="data/dataset_crater_detection_80k.h5", group="test")
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

    images, targets = next(iter(loader))

    out = model(images)
    print(out[0].keys())

    with torch.no_grad():
        torch.onnx.export(
            model,
            (images,),
            "blobs/CraterRCNN.onnx",
            # do_constant_folding=True,
            verbose=False,
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=11,
            do_constant_folding=True,
            input_names=['image_in'],  # the model's input names
            output_names=['boxes', 'labels', 'scores', 'ellipse_matrices'],
            dynamic_axes={'image_in': {0: 'batch'},
                          'boxes': {0: 'sequence'},
                          'labels': {0: 'sequence'},
                          'scores': {0: 'sequence'},
                          'ellipse_matrices': {0: 'sequence'}}
        )

    # Load the ONNX model
    model_onnx = onnx.load("blobs/CraterRCNN.onnx")

    # Check that the IR is well formed
    onnx.checker.check_model(model_onnx, full_check=True)

    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model_onnx.graph))


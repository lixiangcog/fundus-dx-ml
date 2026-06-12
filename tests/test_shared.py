import torch

from shared import CLASS_NAMES, build_resnet18, get_inference_transform
from PIL import Image


def test_class_names_match_imagefolder_order():
    # torchvision ImageFolder assigns indices alphabetically; the saved model's
    # outputs are only meaningful if CLASS_NAMES preserves that order.
    assert CLASS_NAMES == sorted(CLASS_NAMES)
    assert CLASS_NAMES == ['amd', 'cataract', 'diabetic_retinopathy', 'normal']


def test_inference_transform_shape():
    transform = get_inference_transform()
    image = Image.new("RGB", (800, 600), (128, 64, 32))
    tensor = transform(image)
    assert tensor.shape == (3, 224, 224)


def test_build_resnet18_output_dim():
    model = build_resnet18(len(CLASS_NAMES))
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, len(CLASS_NAMES))


def test_saved_weights_match_class_count():
    state_dict = torch.load("best_model.pth", map_location="cpu")
    assert state_dict["fc.weight"].shape[0] == len(CLASS_NAMES)
    model = build_resnet18(len(CLASS_NAMES))
    model.load_state_dict(state_dict)

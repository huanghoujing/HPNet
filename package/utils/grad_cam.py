import torch


def extract_feature(model, inputs, modules=None):
    handles = []
    feat = []
    for m in modules:
        def func(m, i, o): feat.append(o)
        handles.append(m.register_forward_hook(func))
    grad = []
    for m in modules:
        def func(m, gi, go): grad.append(go)
        handles.append(m.register_backward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return feat, grad


class GradCam(object):
    def __init__(self, model, target_layer_names):
        self.model = model

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):


        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

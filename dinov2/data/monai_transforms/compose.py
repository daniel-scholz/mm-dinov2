from monai.transforms.transform import Transform


class MergeTransform(Transform):
    def __init__(self, *transforms: Transform):
        super().__init__()
        self.transforms = transforms

    def __call__(self, input: object) -> dict[str, object]:
        data = {}
        for transform in self.transforms:
            data.update(transform(input))
        return data

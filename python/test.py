from sample_function import camera_params_to_rays
from make_dataset import get_camera_intrinsic_parameter, get_dataset_raw
from constants import DATASET_PATH


def test_camera_params_to_rays():
    param = get_camera_intrinsic_parameter(DATASET_PATH)
    dataset_raw = get_dataset_raw(DATASET_PATH)
    i = 0
    pose = dataset_raw[i]['pose']
    rgb = dataset_raw[i]['rgb']
    o, d = camera_params_to_rays(param, pose)
    print(o.shape, d.shape)
    print(o[0][0], d[0][0])


if __name__ == "__main__":
    test_camera_params_to_rays()

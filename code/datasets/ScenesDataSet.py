from datasets import SceneData
import numpy as np


def dataloader_collate_fn(scenedata_samples):
    """
    Receives a list of SceneData instances, making up the samples of the batch.
    """
    # Trivial collate function - simply return the list:
    return scenedata_samples

class ScenesDataSet:
    def __init__(
        self,
        data_list,
        return_all,
        min_num_views_sampled = 10,
        max_num_views_sampled = 30,
        inplane_rot_aug_max_angle = None,
        tilt_rot_aug_max_angle = None,
    ):
        super().__init__()
        self.data_list = data_list
        self.return_all = return_all
        self.min_num_views_sampled = min_num_views_sampled
        self.max_num_views_sampled = max_num_views_sampled
        self.inplane_rot_aug_max_angle = inplane_rot_aug_max_angle
        self.tilt_rot_aug_max_angle = tilt_rot_aug_max_angle

    def __getitem__(self, item):
        current_data = self.data_list[item]
        if not self.return_all:
            max_sample = min(self.max_num_views_sampled, len(current_data.y))
            if self.min_num_views_sampled >= max_sample:
                sample_fraction = max_sample
            else:
                sample_fraction = np.random.randint(self.min_num_views_sampled, max_sample + 1)
            # NOTE: The GPU "memory leak" fix implemented before may have been unnecessary whenever we train with return_all = False, since we then appear to create temporary SceneData instances here, references to which we do not collect and keep:
            current_data = SceneData.sample_data(current_data, sample_fraction)

        if self.inplane_rot_aug_max_angle is not None or self.tilt_rot_aug_max_angle is not None:
            current_data = SceneData.apply_rotational_homography_aug(
                current_data,
                inplane_rot_aug_max_angle = self.inplane_rot_aug_max_angle,
                tilt_rot_aug_max_angle = self.tilt_rot_aug_max_angle,
            )

        return current_data

    def __len__(self):
        return len(self.data_list)



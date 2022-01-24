from typing import Tuple, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd

from PIL import Image


class CsvParser:
    CLASS_NAMES = ('target',)

    def __init__(self, video_folders_path: str,
                 csv_file: Optional[str] = None, dataframe: Optional[pd.DataFrame] = None):
        """
        Csv Parser for our detector datasets
        :param video_folders_path: path where the video folders are stored
        :param csv_file: csv file containing annotations.
                         If this is not given then ``dataframe`` must not be empty
        :param dataframe: dataframe containing annotations.
                          If this is not given then ``csv_file`` must not be empty
        """
        if csv_file is not None:
            self._df = pd.read_csv(csv_file)  # type:pd.DataFrame
        elif dataframe is not None:
            self._df = dataframe
        else:
            raise Exception('Either "csv_file" or "dataframe" must not be None!')
        self._n_imgs = self._df.shape[0]
        self._video_img_dir = Path(video_folders_path)

    def __enter__(self) -> 'CsvParser':
        return self

    def __exit__(self, type_, value, traceback) -> None:
        pass

    def __len__(self):
        return self._n_imgs

    def _get_img_path(self, video_num: int, frame_num: int):
        """
        Format the image path given its video and frame number
        :return: image path
        """
        return str(self._video_img_dir / f"video_{video_num}" / f"{frame_num}.jpg")

    def _get_row_data(self, row: pd.Series, normalize: bool = True, return_shape: bool = False) \
            -> Union[Tuple[str, np.ndarray], Tuple[str, np.ndarray, Tuple[int, int]]]:
        """
        Get row data: image path and bounding box(es) corners
        :param row: row from which to fetch the data
        :param normalize: whether to normalize the bounding boxes according to the image width and height
        :param return_shape: whether to return the image shape in the output
        :return: image path and its resp. bounding boxes or
                 image path, its resp. bounding boxes and image shape
        """
        image_path = self._get_img_path(row["video_id"], row["video_frame"])
        annotations = eval(row["annotations"])
        # assuming a unique class 0
        bboxes = np.array([(0, ann['x'] + ann['width'] / 2, ann['y'] + ann['height'] / 2, ann['width'], ann['height'])
                           for ann in annotations])
        if normalize or return_shape:
            w, h = Image.open(image_path).size[:2]  # Image.open() is faster than cv2.imread()
            if normalize and len(bboxes) > 0:
                bboxes[:, (1, 3)] /= w
                bboxes[:, (2, 4)] /= h
                bboxes[:, 1:5] = np.clip(bboxes[:, 1:5], 0., 1.)
            if return_shape:
                return image_path, bboxes, (w, h)
        return image_path, bboxes

    def get_row_data(self, index: int, normalize: bool = False, return_shape: bool = False) \
            -> Union[Tuple[str, np.ndarray], Tuple[str, np.ndarray, Tuple[int, int]]]:
        """
        Get row data from index: image path and bounding box(es) corners (and if given the shape)
        :param index: index from the dataframe to fetch
        :param normalize: whether to normalize the bounding boxes according to the image width and height
        :param return_shape: whether to return the image shape in the output
        :return: image path and its resp. bounding boxes or
                 image path, its resp. bounding boxes and image shape
        """
        row = self._df.iloc[index]
        return self._get_row_data(row, normalize, return_shape)

    def read_line(self, normalize: bool = False, return_shape: bool = False) -> Tuple[str, np.ndarray]:
        """
        Generator to read a csv row by row
        :param normalize: whether to normalize the bounding boxes according to the image width and height
        :param return_shape: whether to return the image shape in the output
        :return: image path and its resp. bounding boxes or
                 image path, its resp. bounding boxes and image shape
        """
        for row in self._df.iterrows():
            yield self._get_row_data(row, normalize=normalize, return_shape=return_shape)  # noqa

    def read_all(self, normalize: bool = True, return_shape: bool = True):
        """
        Read all image paths, labels and, if given, image shapes
        :param normalize: whether to normalize the bounding boxes according to the image width and height
        :param return_shape: whether to return the image shape in the output
        :return: image path and its resp. bounding boxes or
                 image path, its resp. bounding boxes and image shape
        """
        img_paths, bboxes, shapes = [], [], []
        for i in range(len(self)):
            outs = self.get_row_data(i, normalize=normalize, return_shape=return_shape)
            img_paths.append(outs[0])
            bboxes.append(outs[1])
            if return_shape:
                shapes.append(outs[2])
        if return_shape:
            return img_paths, bboxes, shapes
        else:
            return img_paths, bboxes

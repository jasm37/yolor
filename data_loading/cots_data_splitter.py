from typing import Optional

import pandas as pd


class COTSDataSplitter:
    def __init__(self, train_perc: float, num_groups: int, cat_name: str = 'video_id',
                 csv_file: Optional[str] = None, dataframe: Optional[pd.DataFrame] = None):
        """
        Helps split a dataframe of sequential images into train and validation sets
        :param train_perc: training set percentage.
                           Assumes that the validation percentage will be ``1 - train_perc``
        :param num_groups: number of season groups to split the dataframe into
        :param cat_name: category name in which to split
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
        self._train_perc = train_perc
        self._train_df = None
        self._val_df = None
        self._split(num_groups, cat_name)

    @property
    def train_df(self):
        return self._train_df

    @property
    def val_df(self):
        return self._val_df

    def reduce_df_sizes(self, n: int = 5):
        """
        Reduce training and validation dataframes size by selecting the first n elements
        This is used to, for example, test new code implementations and make training faster
        :param n: number of elements to reduce the frames to
        """
        self._train_df = self._train_df.head(n=n)
        self._val_df = self._val_df.head(n=n)

    def _split(self, num_groups: int, cat_name: str = 'video_id'):
        """
        Divide the dataframe into parts according to ``cat_name`` and
        then split each part/sequence into train/val sets
        :param num_groups: number of groups to divide each sequence for example:
                           sequence: 1,2,3,4,...20
                           num_groups: 4
                           Then groups = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10],
                                          [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]
        :param cat_name: column name to divide the dataframe

        So, for example if ``train_perc`` is 0.75, ``num_groups`` is 4 and for ``cat_name`` there are two groups:
        A: [1,2,3,4,...20]
        B: [100, 101, .. 200]
        Then the training set would be [1,2,3, .. 14, 15] + [100, 101, .. 148, 149, 150]
        and the validation set would be [16, 17, 18, 19, 20] + [151, 152, 152, .. 199, 200]
        """
        train_idxs = []
        val_idxs_list = []

        def func_num_entries(_str: str):
            # Convert string to struct and get the number of bounding boxes in it
            return len(eval(_str))

        categories = self._df[cat_name].unique()
        for cat in categories:
            cat_df = self._df[self._df[cat_name] == cat]
            num_bboxes_per_frame_df = cat_df['annotations'].apply(func_num_entries)
            num_frames = len(cat_df.index)
            # Assumes that only nonempty frames are needed for training
            nonempty_frames_df = num_bboxes_per_frame_df[num_bboxes_per_frame_df > 0]
            for name, group in nonempty_frames_df.groupby(nonempty_frames_df.index // (num_frames / num_groups)):
                # Get total number of empty frames in the group
                total = group.sum()
                # Compute accumulated sum per group
                cumsum = group.cumsum()
                # Fetch frame indices that belong in the first "train_perc" of the sequence for training
                train_idx_list = cumsum[cumsum <= self._train_perc * total].index.to_list()
                train_idxs += train_idx_list
                # Fetch frame indices that belong in the last "val_perc" of the sequence for validation
                _val_idx_list = cumsum[cumsum > self._train_perc * total].index.to_list()
                # Grab the first row after train_idxs until the last of _val_idx_list
                # This allows to grab rows with no bounding boxes for validation
                val_idxs_list.append([max(train_idx_list) + 1, max(_val_idx_list) + 1])

        self._train_df = self._df.iloc[train_idxs]
        self._val_df = pd.concat([self._df.iloc[start:end] for (start, end) in val_idxs_list], ignore_index=True)

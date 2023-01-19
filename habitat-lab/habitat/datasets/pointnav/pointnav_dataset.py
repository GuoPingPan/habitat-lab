#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
from typing import TYPE_CHECKING, List, Optional

from habitat.config import read_write
from habitat.core.dataset import ALL_SCENES_MASK, Dataset
from habitat.core.registry import registry
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


@registry.register_dataset(name="PointNav-v1")
class PointNavDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads Point Navigation dataset."""

    episodes: List[NavigationEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    @staticmethod
    def check_config_paths_exist(config: "DictConfig") -> bool:
        return os.path.exists(
            config.data_path.format(split=config.split)
        ) and os.path.exists(config.scenes_dir)

    @classmethod
    def get_scenes_to_load(cls, config: "DictConfig") -> List[str]:
        r"""Return list of scene ids for which dataset has separate files with
        episodes.
        """
        dataset_dir = os.path.dirname(
            config.data_path.format(split=config.split)
        )
        if not cls.check_config_paths_exist(config):
            raise FileNotFoundError(
                f"Could not find dataset file `{dataset_dir}`"
            )

        cfg = config.copy()
        with read_write(cfg):
            cfg.content_scenes = []
            dataset = cls(cfg)
            has_individual_scene_files = os.path.exists(
                dataset.content_scenes_path.split("{scene}")[0].format(
                    data_path=dataset_dir
                )
            )
            if has_individual_scene_files:
                return cls._get_scenes_from_folder(
                    content_scenes_path=dataset.content_scenes_path,
                    dataset_dir=dataset_dir,
                )
            else:
                # Load the full dataset, things are not split into separate files
                cfg.content_scenes = [ALL_SCENES_MASK]
                dataset = cls(cfg)
                return list(map(cls.scene_from_scene_path, dataset.scene_ids))

    @staticmethod
    def _get_scenes_from_folder(
        content_scenes_path: str, dataset_dir: str
    ) -> List[str]:
        scenes: List[str] = []

        # {data_path}/content/{scene}.json.gz
        content_dir = content_scenes_path.split("{scene}")[0] # {data_path}/content/
        scene_dataset_ext = content_scenes_path.split("{scene}")[1] # .json.gz
        content_dir = content_dir.format(data_path=dataset_dir)

        # 没有 content 就返回空
        if not os.path.exists(content_dir):
            return scenes

        for filename in os.listdir(content_dir):
            if filename.endswith(scene_dataset_ext):
                scene = filename[: -len(scene_dataset_ext)]
                scenes.append(scene)
        scenes.sort()
        return scenes

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.episodes = []

        if config is None:
            return

        # data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz
        datasetfile_path = config.data_path.format(split=config.split)

        # key: 对于gibson来说这里是空的
        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.scenes_dir)

        # key: 每个场景分开读取
        # Read separate file for each scene

        # data/datasets/pointnav/gibson/v1/{split}/
        dataset_dir = os.path.dirname(datasetfile_path)

        # {data_path}/content/{scene}.json.gz
        has_individual_scene_files = os.path.exists(
            # data/datasets/pointnav/gibson/v1/{split}/content/
            self.content_scenes_path.split("{scene}")[0].format(
                data_path=dataset_dir
            )
        )

        # key: 如果存在 data/datasets/pointnav/gibson/v1/{split}/content/
        if has_individual_scene_files:
            scenes = config.content_scenes

            # key：如果 *，则意味着不指定scenes，全部都读进来
            if ALL_SCENES_MASK in scenes:
                scenes = self._get_scenes_from_folder(
                    content_scenes_path=self.content_scenes_path,
                    dataset_dir=dataset_dir,
                )

            for scene in scenes:
                # {data_path}/content/{scene}.json.gz
                scene_filename = self.content_scenes_path.format(
                    data_path=dataset_dir, scene=scene
                )
                with gzip.open(scene_filename, "rt") as f:
                    self.from_json(f.read(), scenes_dir=config.scenes_dir)

        # key：如果是整个读取的，就需要对所有的episode有效性进行验证
        # data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz
        else:
            self.episodes = list(
                filter(self.build_content_scenes_filter(config), self.episodes)
            )

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        '''

        :param json_str:
        :param scenes_dir:
        :return: self.episodes
        '''
        deserialized = json.loads(json_str)
        # data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for episode in deserialized["episodes"]:
            episode = NavigationEpisode(**episode)

            # key: scenes_dir 在 config.habitat.dataset的作用域下
            # key: default scenes_dir='data/scene_datasets'
            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):

                    # gibson/Adrian.glb
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                # "scene_id": "data/scene_datasets/gibson/Adrian.glb"
                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)
            self.episodes.append(episode)

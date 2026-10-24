import numpy as np
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.models.objects import (BreadObject, BreadVisualObject,
                                      CanObject, CanVisualObject, CerealObject,
                                      CerealVisualObject)


class PickPlaceBreadCereal(PickPlace):

    def __init__(self, **kwargs):
        assert (
            "single_object_mode" not in kwargs and "object_type" not in kwargs
        ), "invalid set of arguments"
        super().__init__(single_object_mode=0, **kwargs)

    def _construct_visual_objects(self):
        self.visual_objects = []
        for vis_obj_cls, obj_name in zip(
            (BreadVisualObject, CerealVisualObject),
            self.obj_names,
        ):
            vis_name = "Visual" + obj_name
            vis_obj = vis_obj_cls(name=vis_name)
            self.visual_objects.append(vis_obj)

    def _construct_objects(self):
        self.objects = []
        for obj_cls, obj_name in zip(
            (BreadObject, CerealObject),
            self.obj_names,
        ):
            obj = obj_cls(name=obj_name)
            self.objects.append(obj)

    def reward(self, action=None):
        """
        Taken from https://github.com/ARISE-Initiative/robosuite/blob/76842f918f16dab5582a128e06b262fc74a70700/robosuite/environments/manipulation/pick_place.py#L265
        Update hardcoding of division by 4.
        """
        # compute sparse rewards
        self._check_success()
        reward = np.sum(self.objects_in_bins)

        # add in shaped rewards
        if self.reward_shaping:
            staged_rewards = self.staged_rewards()
            reward += max(staged_rewards)
        if self.reward_scale is not None:
            reward *= self.reward_scale
            if self.single_object_mode == 0:
                reward /= len(self.objects)
        return reward

    def _check_success(self):
        """
        Remove the r_reach logic as object sizes might be very different
        """
        # remember objects that are in the correct bins
        for i, obj in enumerate(self.objects):
            obj_str = obj.name
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            self.objects_in_bins[i] = int((not self.not_in_bin(obj_pos, i)))

        # returns True if a single object is in the correct bin
        if self.single_object_mode in {1, 2}:
            return np.sum(self.objects_in_bins) > 0

        # returns True if all objects are in correct bins
        return np.sum(self.objects_in_bins) == len(self.objects)


class PickPlaceBreadCan(PickPlace):

    def __init__(self, **kwargs):
        assert (
            "single_object_mode" not in kwargs and "object_type" not in kwargs
        ), "invalid set of arguments"
        super().__init__(single_object_mode=0, **kwargs)

    def _construct_visual_objects(self):
        self.visual_objects = []
        for vis_obj_cls, obj_name in zip(
            (CanVisualObject, BreadVisualObject),
            self.obj_names,
        ):
            vis_name = "Visual" + obj_name
            vis_obj = vis_obj_cls(name=vis_name)
            self.visual_objects.append(vis_obj)

    def _construct_objects(self):
        self.objects = []
        for obj_cls, obj_name in zip(
            (CanObject, BreadObject),
            self.obj_names,
        ):
            obj = obj_cls(name=obj_name)
            self.objects.append(obj)

    def reward(self, action=None):
        """
        Taken from https://github.com/ARISE-Initiative/robosuite/blob/76842f918f16dab5582a128e06b262fc74a70700/robosuite/environments/manipulation/pick_place.py#L265
        Update hardcoding of division by 4.
        """
        # compute sparse rewards
        self._check_success()
        reward = np.sum(self.objects_in_bins)

        # add in shaped rewards
        if self.reward_shaping:
            staged_rewards = self.staged_rewards()
            reward += max(staged_rewards)
        if self.reward_scale is not None:
            reward *= self.reward_scale
            if self.single_object_mode == 0:
                reward /= len(self.objects)
        return reward

    def _check_success(self):
        """
        Remove the r_reach logic as object sizes might be very different
        """
        # remember objects that are in the correct bins
        for i, obj in enumerate(self.objects):
            obj_str = obj.name
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            self.objects_in_bins[i] = int((not self.not_in_bin(obj_pos, i)))

        # returns True if a single object is in the correct bin
        if self.single_object_mode in {1, 2}:
            return np.sum(self.objects_in_bins) > 0

        # returns True if all objects are in correct bins
        return np.sum(self.objects_in_bins) == len(self.objects)

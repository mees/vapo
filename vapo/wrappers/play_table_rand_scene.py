from collections import defaultdict, deque
import logging
from random import shuffle

import numpy as np
from vr_env.scene.play_table_scene import PlayTableScene

logger = logging.getLogger(__name__)


class PlayTableRandScene(PlayTableScene):
    def __init__(self, **args):
        super(PlayTableRandScene, self).__init__(**args)
        # all the objects
        self.obj_names = list(self.object_cfg["movable_objects"].keys())
        self.objs_per_class, self.class_per_obj = {}, {}
        self._find_obj_class()
        self.load_only_one = args["load_only_one"]
        self.counts = None
        if "positions" in args:
            self.rand_positions = args["positions"]
            if self.load_only_one:
                self.counts = {c: deque(maxlen=args["max_counts"]) for c in self.obj_names}
            else:
                self.counts = {c: deque(maxlen=args["max_counts"]) for c in self.objs_per_class.keys()}
        else:
            self.rand_positions = None

        # Load Environment
        self._target = "banana"
        if self.rand_positions:
            self.pick_rand_scene(load=True)
        else:
            self.table_objs = self.obj_names
            self.pick_table_obj()

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        self._target = value

    # Loading random objects
    def _find_obj_class(self):
        obj_names = self.obj_names
        objs_per_class = defaultdict(list)
        class_per_object = {}

        # Find classes in objects
        for name in obj_names:
            if "_" in name:
                # Remove last substring (i.e. class count)
                class_name = "_".join(name.split("_")[:-1])
                objs_per_class[class_name].append(name)
            else:
                objs_per_class["misc"].append(name)

        # Assign objects to classes
        classes = list(objs_per_class.keys())
        for class_name in classes:
            objs_in_class = objs_per_class[class_name]
            if len(objs_in_class) < 1:
                objs_per_class["misc"].extend(objs_in_class)
                class_per_object.update({obj: "misc" for obj in objs_in_class})
                objs_per_class.pop(class_name)
            else:
                class_per_object.update({obj: class_name for obj in objs_in_class})
        self.objs_per_class = objs_per_class
        self.class_per_obj = class_per_object

    def normalize_class_dist(self, decreasing=False):
        probs = {}
        for label, hits in self.counts.items():
            if len(hits) == 0:
                probs[label] = 0
            else:
                probs[label] = np.sum(hits) / len(hits)
        weights = np.array(list(probs.values()))
        w_sum = weights.sum(axis=0)
        if w_sum == 0:
            weights = np.ones_like(weights) * (1 / len(weights))
        else:
            weights = weights / w_sum
            # Make less successful more likely
            if decreasing:
                weights = 1 - weights
                weights = weights / weights.sum(axis=0)
        return probs, weights

    def pick_table_obj(self, eval=False):
        """
        counts = {class: counts}
        """
        if self.load_only_one:
            self.target = self.table_objs[0]
            return

        # More than one obj in table
        if self.counts is not None and not eval:
            _normalized_dist = self.normalize_class_dist(decreasing=True)
            probs, weights = _normalized_dist
            choose_class = self.np_random.choice(list(probs.keys()), p=weights)
            _class_in_table = []
            for obj in self.table_objs:
                if self.class_per_obj[obj] == choose_class:
                    _class_in_table.append(obj)
            self.target = self.np_random.choice(_class_in_table)
        else:
            self.target = self.np_random.choice(self.table_objs)

    def get_scene_with_objects(self, obj_lst, load_scene=False, positions=None):
        """
        obj_lst: list of strings containing names of objs
        load_scene: Only true in initialization of environment
        """
        if positions:
            assert len(obj_lst) <= len(positions)
            rand_pos = positions[: len(obj_lst)]
        else:
            assert len(obj_lst) <= len(self.rand_positions)
            rand_pos = self.rand_positions[: len(obj_lst)]
        # movable_objs is a reference to self.object_cfg
        if load_scene:
            movable_objs = self.object_cfg["movable_objects"]
            # Add positions to new table ojects
            for name, new_pos in zip(obj_lst, rand_pos):
                movable_objs[name]["initial_pos"][:2] = new_pos

            # Add other objects away from view
            far_objs = {k: v for k, v in movable_objs.items() if k not in obj_lst}
            far_pos = [[100 + 20 * i, 0] for i in range(len(far_objs))]
            for i, (name, properties) in enumerate(far_objs.items()):
                movable_objs[name]["initial_pos"][:2] = far_pos[i]
            self.load()
        else:
            movable_objs = {obj.name: i for i, obj in enumerate(self.movable_objects)}
            # Add positions to new table ojects
            for name, new_pos in zip(obj_lst, rand_pos):
                _obj = self.movable_objects[movable_objs[name]]
                _obj.initial_pos[:2] = new_pos

            # Add other objects away from view
            far_objs = {k: v for k, v in movable_objs.items() if k not in obj_lst}
            far_pos = [[100 + 20 * i, 0] for i in range(len(far_objs))]
            for i, (name, properties) in enumerate(far_objs.items()):
                _obj = self.movable_objects[movable_objs[name]]
                _obj.initial_pos[:2] = far_pos[i]

        self.table_objs = obj_lst.copy()

    def choose_new_objs(self, replace_all=False, load_scene=False):
        n_objs = len(self.rand_positions)
        if replace_all:
            # Full random scene
            choose_from = []
            for v in self.objs_per_class.values():
                choose_from.extend(v)

            # At least 1 obj from each class in env
            rand_objs = []
            rand_obj_classes = list(self.objs_per_class.keys())
            for class_name in rand_obj_classes:
                class_objs = self.objs_per_class[class_name]
                rand_obj = self.np_random.choice(class_objs, 1)
                rand_objs.extend(rand_obj)
                choose_from.remove(rand_obj)
                n_objs -= 1
            rand_objs.extend(self.np_random.choice(choose_from, n_objs, replace=False))
        else:
            probs, weights = self.normalize_class_dist()
            rand_objs = []
            for obj in self.table_objs:
                obj_class = self.class_per_obj[obj]
                # Replace obj for another of the same class
                remaining_objs = [o for o in self.objs_per_class[obj_class] if o not in rand_objs and o != obj]

                # If class has 30% success rate change object
                if len(remaining_objs) > 0 and probs[obj_class] > 0.3:
                    rand_obj = self.np_random.choice(remaining_objs)
                    rand_objs.append(rand_obj)
                else:  # keep
                    rand_objs.append(obj)

            # If we have more positions to fill than classes
            if n_objs - len(rand_objs) >= 0:
                n_objs -= len(rand_objs)
            else:
                logger.error("Objects to be placed larger than positions available")
                rand_objs_str = ", ".join(rand_objs)
                logger.info(
                    "random objects (%d): %s, positions available: %d" % (len(rand_objs), rand_objs_str, n_objs)
                )
            rand_objs = list(rand_objs)
            choose_from = [o for o in self.obj_names if o not in rand_objs]
            extra_objs = self.np_random.choice(choose_from, n_objs)
            rand_objs.extend(extra_objs)
            print_str = "Classes in env: \n"
            for obj in rand_objs:
                print_str += "%s: %s \n" % (obj, self.class_per_obj[obj])
            logger.info(print_str)
        shuffle(rand_objs)
        self.get_scene_with_objects(rand_objs, load_scene=load_scene)

    def pick_one_rand_obj(self, load_scene):
        # Get position
        rand_pos = self.object_cfg["fixed_objects"]["table"]["initial_pos"]
        rand_pos = self.np_random.uniform(rand_pos - np.array([0.07, 0.05, 0]), rand_pos + np.array([0.02, 0.05, 0]))[
            :2
        ]
        # Get obj
        # probs, weights = self.normalize_class_dist(self.counts, decreasing=True)
        # obj = self.np_random.choice(list(probs.keys()), p=weights)
        obj = self.np_random.choice(list(self.counts.keys()))
        self.get_scene_with_objects([obj], positions=[rand_pos], load_scene=load_scene)

    def pick_rand_scene(self, objs_success=None, load=False, eval=False):
        # Increment counter
        if objs_success is not None:
            if self.load_only_one:
                for obj, v in objs_success.items():
                    self.counts[obj].append(v)
            else:
                for obj, v in objs_success.items():
                    obj_class = self.class_per_obj[obj]
                    self.counts[obj_class].append(v)

        if self.rand_positions is not None:
            if self.load_only_one and not eval:
                self.pick_one_rand_obj(load_scene=load)
            else:
                replace_all = eval or load
                self.choose_new_objs(replace_all=replace_all, load_scene=load)

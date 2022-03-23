# Usage

## VR Data Collection & Rendering
    vrdatacollector.py
    datarenderer.py
### Record

Run the data collector. It will store pickle files in a directory, normally just named as a time stamp.

Example:

    $ python vrdatacollector.py
    $ python vrdatacollector.py save_dir=<PATH_TO_SAVE_DIR>
    $ python vrdatacollector.py record=false #  test VR env without recording
    $ python vrdatacollector.py hydra.verbose=true #  debug log output, FPS info

In Linux you need to specify the HTC Vive controller:

    $ python vrdatacollector.py record=false vr_controller=vive

For windows you can specify the Oculus controller:

    $ python vrdatacollector.py record=false vr_controller=oculus

You can load different scenes like this:

    $ python vrdatacollector.py scene=basic_playtable
    $ python vrdatacollector.py scene=basic_tabletop
    $ python vrdatacollector.py scene=clutter_playtable
    $ python vrdatacollector.py scene=cluter_tabletop

Speech output

    $ python vrdatacollector.py data_recorder.enable_tts=true


### Render
After collecting some data, point the renderer at the directory of the just recorded files.

Example:

    $ python datarenderer.py load_dir=<PATH_TO_RECORDING>
    $ python datarenderer.py load_dir=<PATH_TO_RECORDING> save_dir=<PATH_TO_SAVE_RENDERING>
    $ python datarenderer.py load_dir=<PATH_TO_RECORDING> save_dir=<PATH_TO_SAVE_RENDERING> camera_conf=static_gripper_opposing #render all three cameras
    $ python datarenderer.py load_dir=<PATH_TO_RECORDING> show_gui=false #  headless rendering
    $ python datarenderer.py load_dir=<PATH_TO_RECORDING> processes=24 #
    $ python datarenderer.py load_dir=<PATH_TO_RECORDING> show_gui=true set_static_cam=true # position static camera before rendering (follow cmd line instructions)
    $ python datarenderer.py load_dir=<PATH_TO_RECORDING> hydra.run.dir=<PATH_TO_EXISTING_RENDERED_EPISODE> # render to folder that already contains rendered episodes

#### Output files
Output is saved in ``*.npz`` (compressed archives that can, e.g., be loaded with ``np.load`` as a dict-like object) that contains the following entries:

| key                    | shape                                    | dtype   | comment                                                                            |   |
|------------------------|------------------------------------------|---------|------------------------------------------------------------------------------------|---|
| ``actions``            | (``n_frames``, 8)                        | float64 | actions for each frame                                                             |   |
| ``actions_labels``     | (8,)                                     | <U14    | description of the columns of ``actions``                                          |   |
| ``robot_obs``          | (``n_frames``, 16)                       | float64 | proprioceptive state observations for each frame                                                  |   |
| ``robot_obs_labels``   | (16,)                                    | <U25    | description of the columns of ``robot_obs``                                        |   |
| ``scene_obs``          | (``n_frames``, 45)                       | float64 | scene state observations for each frame                                                  |   |
| ``scene_obs_labels``   | (45,)                                    | <U25    | description of the columns of ``scene_obs``                                        |   |
| ``frame_times``        | (``n_frames``,)                          | float64 | time of each frame in seconds (since 1970-01-01)                                   |   |
| ``rgb_<cam_name>``     | (``n_frames``, ``height``, ``width``, 3) | uint8   | RGB data of camera <cam_name>                                                      |   |
| ``depth_<cam_name>``   | (``n_frames``, ``height``, ``width``)    | float32 | depth data of camera <cam_name>                                                    |   |
| ``cam_names``          | (``n_cams``,)                            | <U7     | names of all cameras in this rendering                                             |   |
| ``cam_infos``          | ``(n_cams``, 7)                          | float64 | infos on camera position/direction                                                 |   |
| ``cam_infos_labels``   | (7,)                                     | <U14    | description of the entries of ``cam_infos``                                        |   |
| ``obj_pos_orn``        | (``n_frames``, ``n_objects``, 7)         | float64 | positions and orientations of scene objects/"bodies" (including robot)             |   |
| ``obj_pos_orn_labels`` | (7,)                                     | <U15    | description of last dimension of ``obj_pos_orn``                                   |   |
| ``obj_names``          | (``n_objects``,)                         | <U23    | names of the objects in the scenes (describing second dimension of ``obj_pos_orn`` |   |


Generally, entries whose keys end in ``_labels`` are string arrays
describing the entries of the corresponding arrays.  The dtype ``<U``
corresponds to little-endian unicode strings, the exact dtype varies
based on the maximal length of the string values.

This table has been last updated on January 29, 2021; see
``src/datarenderer.py::save_episode`` in case the implementation
should have diverged.

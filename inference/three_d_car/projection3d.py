import bpy, bgl, blf, sys
from bpy import data, ops, props, types, context
import mathutils
import numpy as np
import io


class CaptureImage():
    def __init__(self):
        self.cameraNames = ''
        for arg in sys.argv:
            words = arg.split('=')
            if (words[0] == 'cameras'):
                self.cameraNames = words[1]
        self.sceneKey = bpy.data.scenes.keys()[0]

    def update_camera(self, camera, looking_direction=mathutils.Vector((0.0, -6.5, 1.2)), distance=6.5):
        rot_quat = looking_direction.to_track_quat('Y', 'Z')

        camera.rotation_euler = rot_quat.to_euler()
        camera.location = rot_quat * mathutils.Vector((0.0, 0.0, distance))

    def trans_car(self, list_object, position, angle, scale):
        for obj in bpy.data.objects:
            if (obj.type == 'MESH'):
                obj.location = position
                obj.rotation_euler = angle
                obj.scale = scale

    def capture_image(self, list_object, yaw, pitch):
        for obj in list_object:
            if (obj.type == 'CAMERA') and (self.cameraNames == '' or obj.name.find(self.cameraNames) != -1):
                # Set Scenes camera and output filename
                obj.rotation_euler = (pitch * 3.1415 / 180, 0, 0)
                bpy.data.scenes[self.sceneKey].camera = obj
                bpy.data.scenes[0].render.image_settings.file_format = "PNG"
                bpy.data.scenes[self.sceneKey].render.filepath = 'images//capture_image_' + str(pitch) + "_"+ str(yaw)
                # Render Scene and store the scene
                bpy.ops.render.render(write_still=True)

    def get_image_engine(self, angle):
        self.rotate_car(bpy.data.objects, angle)
        self.capture_image(bpy.data.objects)
        return "images/capture_image.png"


a = CaptureImage()

step = 10
min_pitch = 10
max_pitch = 120

min_yaw = 0
max_yaw = 360

for pitch in range(min_pitch, max_pitch, step):
    for yaw in range(min_yaw, max_yaw, step):
        _pitch = pitch * 3.1415 / 180
        _yaw = yaw * 3.1415 / 180
        car_z = 1.25 - 6.5 * np.tan(3.1415 / 2 - _pitch)
        position = (0, 0, car_z)
        angle = (0, 0, _yaw)

        ratio_scale = abs(car_z) / 10.0 + 0.8
        scale = (ratio_scale, ratio_scale, ratio_scale)
        a.trans_car(bpy.data.objects, position, angle, scale)
        a.capture_image(bpy.data.objects, yaw, pitch)

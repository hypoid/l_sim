import numpy as np
import random
import time
import cv2
from direct.showbase.ShowBase import ShowBase
from panda3d.core import FrameBufferProperties, WindowProperties
from panda3d.core import GraphicsPipe, GraphicsOutput
from panda3d.core import Texture
from panda3d.core import loadPrcFileData

from panda3d.core import Vec3, Point3
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletDebugNode
from panda3d.bullet import BulletSphereShape
from panda3d.bullet import BulletCylinderShape
from panda3d.bullet import XUp
from panda3d.bullet import YUp
from panda3d.bullet import ZUp
from panda3d.bullet import BulletGhostNode
from panda3d.bullet import BulletBoxShape
from panda3d.core import BitMask32
from panda3d.core import AmbientLight
from panda3d.core import DirectionalLight
from panda3d.core import Spotlight
from panda3d.core import AntialiasAttrib
from panda3d.core import GraphicsPipeSelection
from panda3d.core import Camera
from panda3d.core import NodePath
from panda3d.core import TransformState
from panda3d.core import RenderState
from panda3d.core import KeyboardButton
import  memory_profiler

loadPrcFileData('', 'sync-video 0')
if __name__ != '__main__':
    loadPrcFileData('', 'window-type none')
    loadPrcFileData('', 'show-frame-rate-meter false')
else:
    loadPrcFileData('', 'show-frame-rate-meter true')
# loadPrcFileData('', 'state-cache false')
# loadPrcFileData('', 'transform-cache false')



class MyApp(ShowBase):
    def __init__(self, screen_size=84, DEBUGGING=False, human_playable=False):
        ShowBase.__init__(self)
        self.forward_button = KeyboardButton.ascii_key(b'w')
        self.backward_button = KeyboardButton.ascii_key(b's')

        self.fps = 20
        self.human_playable = human_playable
        self.actions = 3
        self.last_frame_start_time = time.time()
        self.action_buffer = [1, 1, 1]
        self.last_teleport_time = 0.0
        self.time_to_teleport = False

        if self.human_playable is False:
            winprops = WindowProperties.size(screen_size, screen_size)
            fbprops = FrameBufferProperties()
            fbprops.set_rgba_bits(8, 8, 8, 0)
            fbprops.set_depth_bits(24)
            self.pipe = GraphicsPipeSelection.get_global_ptr().make_module_pipe('pandagl')
            self.imageBuffer = self.graphicsEngine.makeOutput(
                self.pipe,
                "image buffer",
                1,
                fbprops,
                winprops,
                GraphicsPipe.BFRefuseWindow)


            self.camera = Camera('cam')
            self.cam = NodePath(self.camera)
            self.cam.reparentTo(self.render)

            self.dr = self.imageBuffer.makeDisplayRegion()
            self.dr.setCamera(self.cam)

        self.render.setShaderAuto()
        self.cam.setPos(0.5, 0, 6)
        self.cam.lookAt(0.5, 0, 0)

        # Create Ambient Light
        self.ambientLight = AmbientLight('ambientLight')
        self.ambientLight.setColor((0.2, 0.2, 0.2, 1))
        self.ambientLightNP = self.render.attachNewNode(self.ambientLight)
        self.render.setLight(self.ambientLightNP)

        # Spotlight
        self.light = Spotlight('light')
        self.light.setColor((0.9, 0.9, 0.9, 1))
        self.lightNP = self.render.attachNewNode(self.light)
        self.lightNP.setPos(0, 10, 10)
        self.lightNP.lookAt(0, 0, 0)
        self.lightNP.node().getLens().setFov(40)
        self.lightNP.node().getLens().setNearFar(10, 100)
        self.lightNP.node().setShadowCaster(True, 1024, 1024)
        self.render.setLight(self.lightNP)

        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))

        if DEBUGGING is True:
            debugNode = BulletDebugNode('Debug')
            debugNode.showWireframe(True)
            debugNode.showConstraints(True)
            debugNode.showBoundingBoxes(False)
            debugNode.showNormals(False)
            debugNP = render.attachNewNode(debugNode)
            debugNP.show()
            self.world.setDebugNode(debugNP.node())

        self.finger_speed_mps = 0.0
        self.penalty_applied = False
        self.teleport_cooled_down = True
        self.fps = 20
        self.framecount = 0
        self.reset()


    def reset(self):
        namelist = ['Ground',
                    'Conveyor',
                    'Finger',
                    'Block',
                    'Scrambled Block',
                    'Not Rewardable',
                    'Teleport Me']
        for child in self.render.getChildren():
            for test in namelist:
                if child.node().name == test:
                    self.world.remove(child.node())
                    child.removeNode()
                    break


        # Plane
        self.plane_shape = BulletPlaneShape(Vec3(0, 0, 1), 1)
        self.plane_node = BulletRigidBodyNode('Ground')
        self.plane_node.addShape(self.plane_shape)
        self.plane_np = self.render.attachNewNode(self.plane_node)
        self.plane_np.setPos(0.0, 0.0, -1.0)
        self.world.attachRigidBody(self.plane_node)

        # Conveyor
        self.conv_node = BulletRigidBodyNode('Conveyor')
        self.conv_node.setFriction(1.0)
        self.conv_np = self.render.attachNewNode(self.conv_node)
        self.conv_shape = BulletBoxShape(Vec3(100.0, 1.0, 0.05))
        self.conv_node.setMass(1000.0)
        self.conv_np.setPos(-95.0, 0.0, 0.1)
        self.conv_node.addShape(self.conv_shape)
        self.world.attachRigidBody(self.conv_node)
        self.model = loader.loadModel('assets/conv.egg')
        self.model.flattenLight()
        self.model.reparentTo(self.conv_np)


        # Finger
        self.finger_node = BulletRigidBodyNode('Finger')
        self.finger_node.setFriction(1.0)
        self.finger_np = self.render.attachNewNode(self.finger_node)
        self.finger_shape = BulletCylinderShape(0.1, 0.25, ZUp)
        self.finger_node.setMass(0)
        self.finger_np.setPos(1.8, 0.0, 0.24 + 0.0254*3.5)
        self.finger_node.addShape(self.finger_shape)
        self.world.attachRigidBody(self.finger_node)
        self.model = loader.loadModel('assets/finger.egg')
        self.model.flattenLight()
        self.model.reparentTo(self.finger_np)

        self.blocks = []
        for block_num in range(15):
            new_block = self.spawn_block(Vec3(28, random.uniform(-3, 3), (0.2 * block_num) + 2.0),
                                         2, random.choice([4, 4,
                                                           6]), random.choice([10,
                                                                                     11,
                                                                                     12,
                                                                                     13,
                                                                                     14,
                                                                                     15, 15,
                                                                                     16, 16,
                                                                                     17, 17,
                                                                                     18, 18, 18, 18,
                                                                                     19, 19, 19, 19,
                                                                                     20, 20, 20, 20, 20,
                                                                                     21, 21, 21, 21,
                                                                                     22, 22, 22, 23,
                                                                                     23, 23, 23, 23,
                                                                                     24]))
            # new_block = self.spawn_block(Vec3(18, 0, (0.2 * block_num) + 2.0),
            #                              2, 4, 24)
            self.blocks.append(new_block)

        self.finger_speed_mps = 0.0
        self.penalty_applied = False
        self.teleport_cooled_down = True
        self.fps = 20
        self.framecount = 0
        self.last_teleport_time = 0.0
        self.time_to_teleport = False

        return self.step(1)[0]


    def spawn_block(self, location, z_inches, y_inches, x_inches):
        """
        Spawns a block
        """
        node = BulletRigidBodyNode('Block')
        node.setFriction(1.0)
        block_np = self.render.attachNewNode(node)
        shape = BulletBoxShape(Vec3(0.0254*y_inches, 0.0254*x_inches, 0.0254*z_inches))
        node.setMass(1.0)
        block_np.setPos(location)
        block_np.setHpr(random.uniform(-60, 60), 0.0, 0.0)
        node.addShape(shape)
        self.world.attachRigidBody(node)
        model = loader.loadModel('assets/bullet-samples/models/box.egg')
        model.setH(90)
        model.setSx(0.0254*x_inches*2)
        model.setSy(0.0254*y_inches*2)
        model.setSz(0.0254*z_inches*2)
        model.flattenLight()
        model.reparentTo(block_np)
        block_np.node().setTag('scrambled', 'False')
        return block_np

    def get_camera_image(self, requested_format=None):
        """
        Returns the camera's image, which is of type uint8 and has values
        between 0 and 255.
        The 'requested_format' argument should specify in which order the
        components of the image must be. For example, valid format strings are
        "RGBA" and "BGRA". By default, Panda's internal format "BGRA" is used,
        in which case no data is copied over.
        """
        tex = self.dr.getScreenshot()
        if requested_format is None:
            data = tex.getRamImage()
        else:
            data = tex.getRamImageAs(requested_format)
        image = np.frombuffer(data, np.uint8)  # use data.get_data() instead of data in python 2
        image.shape = (tex.getYSize(), tex.getXSize(), tex.getNumComponents())
        image = np.flipud(image)
        return image[:,:,:3]

    def reset_conv(self):
        conveyor_dist_left = 1 - self.conv_np.getPos()[0]
        if conveyor_dist_left < 10:
            self.conv_np.setX(-95.0)
            self.conv_np.setY(0.0)


    def check_rewards(self):
        reward = 0
        for block in self.blocks:
            block_x, block_y, block_z = block.getPos()
            if block_z > 0.16 and block_x > -1 and block_x < 0:
                block.node().setTag('scrambled', 'True')
            if block_x < 2.3 and block_z < 0.16 and block.node().getTag('scrambled') == 'True':
                block.node().setTag('scrambled', 'False')
                reward = 1

        return reward

    def check_teleportable(self, blocks_per_minute):
        self.time = self.framecount/self.fps

        # if self.time % (1/(blocks_per_minute/60)) < 0.1:
        #     self.time_to_teleport = True
        # else:
        #     self.time_to_teleport = False
        #     self.teleport_cooled_down = True
        teleport_cooled_down = True if self.last_teleport_time + 0.4 < self.time else False
        if random.choice([True,
                          False, False, False]) and teleport_cooled_down:
            self.last_teleport_time = self.time
            self.time_to_teleport = True

        for block in self.blocks:
            block_x = block.getPos()[0]
            if block_x > 5:
                block.node().setTag('scrambled', 'False')
                if self.time_to_teleport is True:
                    self.time_to_teleport = False
                    block.setX(-3)
                    block.setY(0.0)
                    block.setZ(2.0)
                    block.setHpr(random.uniform(-60, 60), 0.0, 0.0)

    def step(self, action):
        dt = 1/self.fps
        self.framecount += 1
        max_dist = 1.1
        # Move finger
        finger_max_speed = 2
        finger_accel = 10.0
        finger_deccel = 10.0
        self.action_buffer.pop(0)
        self.action_buffer.append(action)
        action = self.action_buffer[0]


        if action == 0:
            self.finger_speed_mps += dt * finger_accel
            if self.finger_speed_mps > finger_max_speed:
                self.finger_speed_mps = 2
        if action == 1:
            if self.finger_speed_mps > 0.01:
                self.finger_speed_mps -= finger_deccel * dt
            if self.finger_speed_mps < -0.01:
                self.finger_speed_mps += finger_deccel * dt
        if action == 2:
            self.finger_speed_mps -= dt * finger_accel
            if self.finger_speed_mps < -finger_max_speed:
                self.finger_speed_mps = -finger_max_speed

        real_displacement = self.finger_speed_mps * dt
        self.finger_np.setY(self.finger_np.getY() + real_displacement)

        if self.finger_np.getY() > max_dist:
            self.finger_np.setY(max_dist)
            self.finger_speed_mps = 0
        if self.finger_np.getY() < -max_dist:
            self.finger_np.setY(-max_dist)
            self.finger_speed_mps = 0


        # self.world.doPhysics(dt, 5, 1.0/120.0)
        self.world.doPhysics(dt, 20, 1.0/240.0)
        self.reset_conv()
        self.check_teleportable(blocks_per_minute=59)

        # Keep the conveyor moving
        self.conv_np.node().setLinearVelocity(Vec3(1.0, 0.0, 0.0))

        if self.human_playable is False:
            self.graphicsEngine.renderFrame()
            TransformState.garbageCollect()
            RenderState.garbageCollect()
            image = self.get_camera_image()
        else:
            image = None

        score = 0
        score += self.check_rewards()
        done = False

        return image, score, done

    def update(self, task):
        is_down = self.mouseWatcherNode.is_button_down
        next_act = 1
        if is_down(self.forward_button):
            next_act = 0
        if is_down(self.backward_button):
            next_act = 2
        _, reward, _ = self.step(next_act)
        if reward != 0:
            print(reward)
        last_frame_duration = time.time() - self.last_frame_start_time
        if last_frame_duration < (1/self.fps):
            time.sleep((1/self.fps) - last_frame_duration)
        self.last_frame_start_time = time.time()
        return task.cont


if __name__ == '__main__':
    app = MyApp(screen_size=84*8, human_playable=True)
    app.taskMgr.add(app.update, 'update')
    app.run()

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

loadPrcFileData('', 'show-frame-rate-meter true')
loadPrcFileData('', 'sync-video 0')



class MyApp(ShowBase):
    def __init__(self, screen_size=84, DEBUGGING=False):
        ShowBase.__init__(self)
        self.render_stuff = True
        self.actions = 3

        self.render.setShaderAuto()
        self.cam.setPos(0, 0, 7)
        self.cam.lookAt(0, 0, 0)

        wp = WindowProperties()
        window_size = screen_size
        wp.setSize(window_size, window_size)
        self.win.requestProperties(wp)

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

        # Reward zone
        self.rzone_shape = BulletBoxShape(Vec3(.8, 1, 0.5))
        self.rzone_ghost = BulletGhostNode('Reward Zone')
        self.rzone_ghost.addShape(self.rzone_shape)
        self.rzone_ghostNP = self.render.attachNewNode(self.rzone_ghost)
        self.rzone_ghostNP.setPos(2.2, 0.0, 0.86)
        self.rzone_ghostNP.setCollideMask(BitMask32(0x0f))
        self.world.attachGhost(self.rzone_ghost)

        # Needed for camera image
        self.dr = self.camNode.getDisplayRegion(0)

        # Needed for camera depth image
        winprops = WindowProperties.size(self.win.getXSize(), self.win.getYSize())
        fbprops = FrameBufferProperties()
        fbprops.setDepthBits(1)
        self.depthBuffer = self.graphicsEngine.makeOutput(
            self.pipe, "depth buffer", -2,
            fbprops, winprops,
            GraphicsPipe.BFRefuseWindow,
            self.win.getGsg(), self.win)
        self.depthTex = Texture()
        self.depthTex.setFormat(Texture.FDepthComponent)
        self.depthBuffer.addRenderTexture(self.depthTex,
            GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth)
        lens = self.cam.node().getLens()
        # the near and far clipping distances can be changed if desired
        # lens.setNear(5.0)
        # lens.setFar(500.0)
        self.depthCam = self.makeCamera(self.depthBuffer,
            lens=lens,
            scene=render)
        self.depthCam.reparentTo(self.cam)

    def reset(self):
        namelist = ['Ground',
                    'Conveyor',
                    'Finger',
                    'Block',
                    'Scrambled Block',
                    'Not Rewardable',
                    'Teleport Me']
        for child in render.getChildren():
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
            new_block = self.spawn_block(Vec3(18, 0, (0.2 * block_num) + 2.0))
            self.blocks.append(new_block)

        self.have_scramble = False
        self.penalty_applied = False
        self.spawnned = False
        self.score = 10
        self.teleport_cooled_down = True
        self.fps = 20
        self.framecount = 0

        return self.step(1)[0]


    def spawn_block(self, location):
        node = BulletRigidBodyNode('Block')
        node.setFriction(1.0)
        block_np = self.render.attachNewNode(node)
        block_np.setAntialias(AntialiasAttrib.MMultisample)
        shape = BulletBoxShape(Vec3(0.0254*4, 0.0254*24, 0.0254*2))
        node.setMass(1.0)
        #block_np.setPos(-3.7, 0.0, 2.0)
        block_np.setPos(location)
        block_np.setHpr(random.uniform(-60, 60), 0.0, 0.0)
        node.addShape(shape)
        self.world.attachRigidBody(node)
        model = loader.loadModel('assets/bullet-samples/models/box.egg')
        model.setH(90)
        model.setSy(0.0254*4*2)
        model.setSx(0.0254*24*2)
        model.setSz(0.0254*2*2)
        model.flattenLight()
        model.reparentTo(block_np)
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
        # self.conv_np.setY(0.0)
        # self.conv_np.setHpr(0.0, 0.0, 0.0)

    def check_penalty(self):
        penalty = 0
        self.pzone_ghost = self.pzone_ghostNP.node()
        for node in self.pzone_ghost.getOverlappingNodes():
            if node.name == 'Block':
                penalty = 1
                node.name = 'Scramble'
                self.have_scramble = False
        return penalty

    def check_rewards(self):
        reward = 0
        # Check for reward blocks (recently cleared scrambles)
        rzone_ghost = self.rzone_ghostNP.node()
        scrambled = False
        for node in rzone_ghost.getOverlappingNodes():
            if node.name == 'Block' or node.name == 'Scrambled Block':
                node.name = 'Scrambled Block'
                scrambled = True

        # Rename blocks that are not eligable for reward due to being too late
        for block in self.blocks:
            block_x = block.getPos()[0]
            block_name = block.node().name
            if block_x > 2.4 and block_name == 'Scrambled Block':
                self.have_scramble = False
                scrambled = False
                block.node().name = 'Not Rewardable'

        if scrambled is True:
            self.have_scramble = True
        else:
            if self.have_scramble is True:
                reward = 1
                self.have_scramble = False
        return reward

    def check_teleportable(self, blocks_per_minute):
        self.time = self.framecount/self.fps
        if self.time % (1/(blocks_per_minute/60)) < 0.1:
            self.time_to_teleport = True
        else:
            self.time_to_teleport = False
            self.teleport_cooled_down = True
        for block in self.blocks:
            block_x = block.getPos()[0]
            if block_x > 5:
                if block.node().name == 'Scrambled Block':
                    self.have_scramble = False
                block.node().name = 'Teleport Me'
                if self.time_to_teleport is True and self.teleport_cooled_down is True:
                    self.teleport_cooled_down = False
                    block.setX(-4)
                    block.setY(0.0)
                    block.setZ(2.0)
                    block.setHpr(random.uniform(-60, 60), 0.0, 0.0)
                    block.node().name = 'Block'

    def step(self, action):
        dt = 1/self.fps
        self.framecount += 1
        finger_meters_per_second = 2
        max_dist = 1.1
        real_displacement = finger_meters_per_second * dt
        # Move finger
        if action == 0:
            self.finger_np.setY(self.finger_np.getY() + real_displacement)
            if self.finger_np.getY() > max_dist:
                self.finger_np.setY(max_dist)

        if action == 2:
            self.finger_np.setY(self.finger_np.getY() - real_displacement)
            if self.finger_np.getY() < -max_dist:
                self.finger_np.setY(-max_dist)

        self.world.doPhysics(dt, 5, 1.0/120.0)
        self.reset_conv()
        self.check_teleportable(blocks_per_minute=1.1*60)

        # Keep the conveyor moving
        self.conv_np.node().setLinearVelocity(Vec3(1.0, 0.0, 0.0))

        if self.render_stuff == True:
            self.graphicsEngine.renderFrame()
        image = self.get_camera_image()
        # image = cv2.resize(image, (84, 84), interpolation=cv2.INTER_CUBIC)

        score = 0
        score += self.check_rewards()
        #score -= self.check_penalty()
        done = False

        return image, score, done




def main():
    cv2.namedWindow('state', flags=cv2.WINDOW_NORMAL)
    app = MyApp(screen_size=84*8)
    num_episodes = 5000
    max_epLength = 10000
    app.reset()
    for ep_number in range(num_episodes):
        for i in range(100):
            image,s,done = app.step(0)
        score = 0
        f = 0
        while True:
            # app.render_stuff = False
            f += 1
            cv2.imshow('state', image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("Pressed ESC or q, exiting")
                exit()
            elif key == 119:
                next_act = 0
            elif key == 115:
                next_act = 2
            else:
                next_act = 1

            image, s, done = app.step(next_act)
            score += s
            if f > max_epLength:
                break
        print((ep_number+1)*max_epLength, score)

if __name__ == '__main__':
    main()

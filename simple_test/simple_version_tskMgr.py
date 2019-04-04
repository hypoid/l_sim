import numpy as np
import time
from direct.showbase.ShowBase import ShowBase
from panda3d.core import PandaNode
from panda3d.core import Vec3
from panda3d.core import Spotlight
from panda3d.core import Camera
from panda3d.core import NodePath


SHOW_IMAGE = False
SHOW_MEM_STATS = True

if SHOW_IMAGE:
    import cv2

if SHOW_MEM_STATS:
    import  memory_profiler

class MyApp(ShowBase):
    def __init__(self, screen_size=84):
        ShowBase.__init__(self, windowType='offscreen')
        self.render.setShaderAuto()

        # Spotlight
        self.light = Spotlight('light')
        self.lightNP = self.render.attachNewNode(self.light)
        self.lightNP.setPos(0, 10, 10)
        self.lightNP.lookAt(0, 0, 0)
        self.render.setLight(self.lightNP)

        # Block
        node = PandaNode('Block')
        block_np = self.render.attachNewNode(node)
        model = loader.loadModel('box.egg')
        model.reparentTo(block_np)

        self.start_time = time.time()
        if SHOW_MEM_STATS:
            self.last_mem = memory_profiler.memory_usage()[0]
            self.step_num = 0


    def get_camera_image(self, requested_format=None):
        dr = self.camNode.getDisplayRegion(0)
        tex = dr.getScreenshot()
        data = tex.getRamImage()
        image = np.frombuffer(data, np.uint8)
        image.shape = (tex.getYSize(), tex.getXSize(), tex.getNumComponents())
        return image


    def rotate_light(self):
        radius = 10
        step = 0.1
        t = time.time() - self.start_time
        angleDegrees = t * step
        angleRadians = angleDegrees * (np.pi / 180.0)
        self.lightNP.setPos(radius * np.sin(angleRadians), -radius * np.cos(angleRadians), 3)
        self.lightNP.lookAt(0, 0, 0)


    def step(self, task):
        self.rotate_light()
        image = self.get_camera_image()
        if SHOW_MEM_STATS:
            self.step_num += 1
            if self.step_num % 1000 == 0:
                now_mem = memory_profiler.memory_usage()[0]
                if self.step_num != 1000:
                    print('Memory per 1k steps:', now_mem-self.last_mem, 'MB')
                self.last_mem = now_mem


        if SHOW_IMAGE:
            cv2.imshow('state', image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("Pressed ESC or q, exiting")
                exit()

        return task.cont


def main():
    app = MyApp(screen_size=84*1)
    app.taskMgr.add(app.step, 'step')
    app.run()

if __name__ == '__main__':
    main()

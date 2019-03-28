from panda3d.core import Vec3
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletBoxShape
from panda3d.core import NodePath


class MyApp():
    def __init__(self):
        self.render = NodePath('render')

        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))

        node = BulletRigidBodyNode('Block')
        block_np = self.render.attachNewNode(node)
        shape = BulletBoxShape(Vec3(0.0254*4, 0.0254*24, 0.0254*2))
        node.setMass(1.0)
        node.addShape(shape)
        self.world.attachRigidBody(node)


def main():
    app = MyApp()
    while True:
        app.world.doPhysics(0.05, 5, 1.0/120.0)

if __name__ == '__main__':
    main()

import random
import inspect

from direct.showbase.InputStateGlobal import inputState
import direct.directbase.DirectStart
from direct.interval.MetaInterval import Sequence
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


random.seed(512)
DEBUGGING = False
if DEBUGGING is True:
    debugNode = BulletDebugNode('Debug')
    debugNode.showWireframe(True)
    debugNode.showConstraints(True)
    debugNode.showBoundingBoxes(False)
    debugNode.showNormals(False)
    debugNP = render.attachNewNode(debugNode)
    debugNP.show()

base.cam.setPos(0, 0, 7)
base.cam.lookAt(0, 0, 0)
base.setFrameRateMeter(True)

# World
world = BulletWorld()
world.setGravity(Vec3(0, 0, -9.81))
if DEBUGGING is True:
    world.setDebugNode(debugNP.node())

# Create Ambient Light
ambientLight = AmbientLight('ambientLight')
ambientLight.setColor((0.2, 0.2, 0.2, 1))
ambientLightNP = render.attachNewNode(ambientLight)
render.setLight(ambientLightNP)


# Plane
plane_shape = BulletPlaneShape(Vec3(0, 0, 1), 1)
plane_node = BulletRigidBodyNode('Ground')
plane_node.addShape(plane_shape)
plane_np = render.attachNewNode(plane_node)
plane_np.setPos(0.0, 0.0, -1.0)
world.attachRigidBody(plane_node)

# Conveyor
conv_node = BulletRigidBodyNode('Conveyor')
conv_node.setFriction(1.0)
conv_np = render.attachNewNode(conv_node)
conv_shape = BulletBoxShape(Vec3(100.0, 1.0, 0.05))
conv_node.setMass(1000.0)
conv_np.setPos(-95.0, 0.0, 0.1)
conv_node.addShape(conv_shape)
world.attachRigidBody(conv_node)
model = loader.loadModel('conv.egg')
model.flattenLight()
model.reparentTo(conv_np)

# Finger
finger_node = BulletRigidBodyNode('Finger')
finger_node.setFriction(1.0)
finger_np = render.attachNewNode(finger_node)
finger_shape = BulletCylinderShape(0.1, 0.25, ZUp)
finger_node.setMass(0)
finger_np.setPos(1.8, 0.0, 0.24 + 0.0254*3.5)
finger_node.addShape(finger_shape)
world.attachRigidBody(finger_node)
model = loader.loadModel('finger.egg')
model.flattenLight()
model.reparentTo(finger_np)

# Spotlight
light = Spotlight('light')
light.setColor((0.9, 0.9, 0.9, 1))
lightNP = render.attachNewNode(light)
# This light is facing backwards, towards the camera.
lightNP.setHpr(180, -20, 0)
lightNP.setPos(0, 10, 10)
lightNP.lookAt(0, 0, 0)
lightNP.node().getLens().setFov(40)
lightNP.node().getLens().setNearFar(10, 100)
#lightNP.node().setShadowCaster(True)
lightNP.node().setShadowCaster(True, 1024, 1024)
# Use a 512x512 resolution shadow map
render.setLight(lightNP)
# Enable the shader generator for the receiving nodes
render.setShaderAuto()

# Delete zone
dzone_shape = BulletBoxShape(Vec3(1, 10, 10))
dzone_ghost = BulletGhostNode('Delete Zone')
dzone_ghost.addShape(dzone_shape)
dzone_ghostNP = render.attachNewNode(dzone_ghost)
dzone_ghostNP.setPos(5.7, 0, 0.0)
dzone_ghostNP.setCollideMask(BitMask32(0x0f))
world.attachGhost(dzone_ghost)

# Penalty zone 1
pzone_shape = BulletBoxShape(Vec3(1, 1, 0.5))
pzone_ghost = BulletGhostNode('Penalty Zone 1')
pzone_ghost.addShape(pzone_shape)
pzone_ghostNP = render.attachNewNode(pzone_ghost)
pzone_ghostNP.setPos(4.2, 0, 0.86)
pzone_ghostNP.setCollideMask(BitMask32(0x0f))
world.attachGhost(pzone_ghost)

# Reward zone
rzone_shape = BulletBoxShape(Vec3(.8, 1, 0.5))
rzone_ghost = BulletGhostNode('Penalty Zone 2')
rzone_ghost.addShape(rzone_shape)
rzone_ghostNP = render.attachNewNode(rzone_ghost)
rzone_ghostNP.setPos(2.5, 0.0, 0.86)
rzone_ghostNP.setCollideMask(BitMask32(0x0f))
world.attachGhost(rzone_ghost)

def spawn_block():
    global world
    node = BulletRigidBodyNode('Block')
    node.setFriction(1.0)
    block_np = render.attachNewNode(node)
    shape = BulletBoxShape(Vec3(0.0254*4, 0.0254*24, 0.0254*2))
    node.setMass(1.0)
    block_np.setPos(-3.7, 0.0, 2.0)
    block_np.setHpr(random.uniform(-60, 60), 0.0, 0.0)
    node.addShape(shape)
    world.attachRigidBody(node)
    model = loader.loadModel('bullet-samples/models/box.egg')
    model.setH(90)
    model.setSy(0.0254*4*2)
    model.setSx(0.0254*24*2)
    model.setSz(0.0254*2*2)
    model.flattenLight()
    model.reparentTo(block_np)


# Update
def update(task):
    global spawnned
    global score
    global was_scrambled

    dt = globalClock.getDt()
    world.doPhysics(dt, 5, 1.0/120.0)

    # Reset the conveyor if need be
    conveyor_dist_left = 1 - conv_np.getPos()[0]
    if conveyor_dist_left < 10:
        conv_np.setX(-95.0)
        conv_np.setY(0.0)
    # conv_np.setY(0.0)
    # conv_np.setHpr(0.0, 0.0, 0.0)


    # Spawn Blocks
    time = task.time
    if int(time * 1.5) % 2 == 0:
        if spawnned is False:
            spawnned = True
            spawn_block()
    else:
        spawnned = False

    # Check for penalty blocks in zone 1
    pzone_ghost = pzone_ghostNP.node()
    for node in pzone_ghost.getOverlappingNodes():
        if node.name == 'Block':
            score -= 1
            # world.remove(node)
            print(score)
            was_scrambled = False
            node.name = 'Scramble'

    # Check for reward blocks (recently cleared scrambles)
    rzone_ghost = rzone_ghostNP.node()
    scrambled = False
    for node in rzone_ghost.getOverlappingNodes():
        if node.name == 'Block':
            scrambled = True
    if scrambled is True:
        was_scrambled = True
    else:
        if was_scrambled is True:
            score += 1
            print(score)
            was_scrambled = False

    # Check for deletable blocks
    dzone_ghost = dzone_ghostNP.node()
    for node in dzone_ghost.getOverlappingNodes():
        if node.name == 'Block' or node.name == 'Scramble':
            world.remove(node)

    # Move finger
    finger_meters_per_second = 2
    real_displacement = finger_meters_per_second * dt
    if inputState.isSet('forward'):
        finger_np.setY(finger_np.getY() + real_displacement)
    if inputState.isSet('reverse'):
        finger_np.setY(finger_np.getY() - real_displacement)

    # Keep the conveyor moving
    conv_np.node().setLinearVelocity(Vec3(0.5, 0.0, 0.0))

    return task.cont

was_scrambled = False
penalty_applied = False
spawnned = False
score = 10
inputState.watchWithModifiers('forward', 'w')
inputState.watchWithModifiers('reverse', 's')
taskMgr.add(update, 'update')
run()

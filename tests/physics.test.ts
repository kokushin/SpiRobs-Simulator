import assert from 'node:assert/strict';
import test from 'node:test';
import * as CANNON from 'cannon-es';
import {
  autoGraspCommand,
  DEFAULT_SETTINGS,
  SpiRobPhysics,
  UNIT_DATA,
} from '../src/physics.ts';
import type { ObjectKind } from '../src/physics.ts';

test('auto-grasp follows the paper antagonistic cable sequence', () => {
  const pack = autoGraspCommand(0.6, 0);
  assert.equal(pack.phase, 'packing');
  assert.deepEqual(pack.cableForces, [3, 0, 0]);

  const reach = autoGraspCommand(3.0, 0);
  assert.equal(reach.phase, 'reaching');
  assert.equal(reach.cableForces[0], 6);
  assert.ok(reach.cableForces[1] > 0 && reach.cableForces[1] < reach.cableForces[0]);
  assert.equal(reach.cableForces[1], reach.cableForces[2]);

  const wrap = autoGraspCommand(6.4, 0);
  assert.equal(wrap.phase, 'wrapping');
  assert.ok(wrap.cableForces[0] < reach.cableForces[0], 'packing side must relax during climbing');
  assert.equal(wrap.cableForces[1], 5.82, 'opposing side stays fixed during climbing');
  assert.equal(wrap.cableForces[1], wrap.cableForces[2]);
  assert.ok(wrap.wrapProgress > 0 && wrap.wrapProgress < 1);

  const grasp = autoGraspCommand(9.3, 0);
  assert.equal(grasp.phase, 'grasping');
  assert.ok(grasp.cableForces[1] > wrap.cableForces[1], 'opposing side tightens the grasp');

  const hold = autoGraspCommand(10.3, 2);
  assert.equal(hold.phase, 'holding');
  assert.equal(hold.cableForces[2], 5.2);
  assert.equal(hold.releaseTarget, true);
});

function runGrasp(objectKind: ObjectKind) {
  const settings = structuredClone(DEFAULT_SETTINGS);
  settings.objectKind = objectKind;
  const physics = new SpiRobPhysics(settings);
  const object = physics.objectBody!;
  object.type = CANNON.Body.KINEMATIC;
  object.position.set(-0.03, -0.06, 0);
  object.updateMassProperties();
  const stagedPosition = object.position.clone();

  for (let step = 0; step < 12 * 120; step++) {
    const command = autoGraspCommand(step / 120, 0);
    settings.cableForces = command.cableForces;
    physics.wrapProgress = command.wrapProgress;
    if (command.releaseTarget && object.type !== CANNON.Body.DYNAMIC) {
      object.type = CANNON.Body.DYNAMIC;
      object.mass = settings.objectMass;
      object.velocity.setZero();
      object.angularVelocity.setZero();
      object.updateMassProperties();
    }
    physics.step(1 / 120);
  }

  return { physics, object, stagedPosition };
}

for (const objectKind of ['sphere', 'cylinder', 'box'] as const) {
  test(`auto-grasp wraps and holds a ${objectKind} without penetration`, () => {
    const { physics, object, stagedPosition } = runGrasp(objectKind);

    assert.equal(physics.isGrasping, true);
    assert.ok(physics.contactingUnits.length >= 3);
    assert.ok(physics.graspQuality >= 0.6);
    assert.ok(object.position.distanceTo(stagedPosition) < 0.004, 'released load should remain held');

    physics.bodies.slice(1).forEach((body, index) => {
      const unit = UNIT_DATA[index + 1];
      const unitRadius = Math.min(unit[2], unit[3]) * 0.001 * 0.43;
      assert.ok(
        physics.objectSignedDistance(body.position, unitRadius) >= -0.001,
        `unit ${index + 2} penetrated the ${objectKind}`,
      );
    });
  });
}

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

interface GraspTelemetry {
  /** Highest unit-center speed over the whole run [m/s]. Teleporting wrap
   *  targets show up here as multi-m/s spikes (dt = 1/120 kinematic drive). */
  maxUnitSpeed: number;
  /** Highest speed observed only while the wrap command is active [m/s]. */
  maxWrapUnitSpeed: number;
  /** Deepest unit-into-object penetration over the whole run [m]. */
  maxPenetration: number;
  /** Largest single-substep displacement of any unit center [m]. */
  maxUnitJump: number;
}

function runGrasp(objectKind: ObjectKind) {
  const settings = structuredClone(DEFAULT_SETTINGS);
  settings.objectKind = objectKind;
  const physics = new SpiRobPhysics(settings);
  const object = physics.objectBody!;
  object.type = CANNON.Body.KINEMATIC;
  object.position.set(-0.03, -0.06, 0);
  object.updateMassProperties();
  const stagedPosition = object.position.clone();

  const telemetry: GraspTelemetry = {
    maxUnitSpeed: 0,
    maxWrapUnitSpeed: 0,
    maxPenetration: 0,
    maxUnitJump: 0,
  };
  let previousPositions = physics.bodies.map((body) => body.position.clone());

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

    physics.bodies.forEach((body, index) => {
      if (index === 0) return;
      const speed = body.velocity.length();
      telemetry.maxUnitSpeed = Math.max(telemetry.maxUnitSpeed, speed);
      if (command.wrapProgress > 0) {
        telemetry.maxWrapUnitSpeed = Math.max(telemetry.maxWrapUnitSpeed, speed);
      }
      const jump = body.position.distanceTo(previousPositions[index]);
      telemetry.maxUnitJump = Math.max(telemetry.maxUnitJump, jump);
      const unit = UNIT_DATA[index];
      const unitRadius = Math.min(unit[2], unit[3]) * 0.001 * 0.43;
      const distance = physics.objectSignedDistance(body.position, unitRadius);
      telemetry.maxPenetration = Math.max(telemetry.maxPenetration, -Math.min(0, distance));
    });
    previousPositions = physics.bodies.map((body) => body.position.clone());
  }

  return { physics, object, stagedPosition, telemetry };
}

for (const objectKind of ['sphere', 'cylinder', 'box'] as const) {
  test(`auto-grasp wraps and holds a ${objectKind} without penetration`, (t) => {
    const { physics, object, stagedPosition, telemetry } = runGrasp(objectKind);
    t.diagnostic(`${objectKind} maxUnitSpeed=${telemetry.maxUnitSpeed.toFixed(3)} m/s`);
    t.diagnostic(`${objectKind} maxUnitJump=${(telemetry.maxUnitJump * 1000).toFixed(2)} mm/substep`);
    t.diagnostic(`${objectKind} maxPenetration=${(telemetry.maxPenetration * 1000).toFixed(3)} mm`);

    // Smoothness: a wrap-target discontinuity (anchor flip, start-unit jump)
    // appears as a multi-m/s kinematic velocity spike. Healthy runs stay
    // below ~1.5 m/s, broken ones historically hit 9-14 m/s.
    assert.ok(
      telemetry.maxWrapUnitSpeed < 2.0,
      `wrap judder: unit speed spiked to ${telemetry.maxWrapUnitSpeed.toFixed(2)} m/s (limit 2.0)`,
    );
    assert.ok(
      telemetry.maxUnitJump < 0.015,
      `wrap teleport: unit jumped ${(telemetry.maxUnitJump * 1000).toFixed(1)} mm in one substep (limit 15 mm)`,
    );
    // Non-penetration must hold for the WHOLE run, not just the final state.
    assert.ok(
      telemetry.maxPenetration < 0.0005,
      `tunneling: unit sank ${(telemetry.maxPenetration * 1000).toFixed(2)} mm into the ${objectKind} (limit 0.5 mm)`,
    );

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

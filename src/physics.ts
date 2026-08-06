import * as CANNON from 'cannon-es';

export const UNIT_DATA = [
  [-165.86769, 13.86136, 27.77303, 31.41718],
  [-183.25348, 19.05084, 26.06857, 29.34485],
  [-202.49631, 17.70167, 24.23258, 27.27514],
  [-220.37899, 16.44785, 22.52637, 25.3518],
  [-236.99764, 15.28265, 20.94077, 23.56439],
  [-252.44162, 14.19982, 19.46723, 21.90326],
  [-266.79398, 13.19355, 18.09785, 20.35956],
  [-280.13184, 12.25836, 16.82526, 18.92499],
  [-292.52693, 11.38931, 15.64262, 17.5918],
  [-304.04588, 10.58169, 14.54358, 16.35272],
  [-314.75065, 9.83115, 13.52222, 15.20136],
  [-324.69876, 9.13366, 12.57306, 14.13122],
  [-333.94368, 8.48544, 11.69098, 13.13678],
  [-342.53516, 7.88312, 10.87126, 12.21265],
  [-350.51935, 7.3233, 10.10946, 11.35376],
  [-357.9392, 6.80307, 9.40152, 10.55554],
  [-364.83454, 6.31961, 8.74361, 9.81372],
  [-371.24255, 5.87036, 8.13221, 9.12433],
  [-377.1976, 5.45282, 7.56401, 8.4837],
  [-382.73171, 5.06479, 7.03599, 7.88824],
] as const;

export type Mount = 'horizontal' | 'hanging' | 'upright';
export type ObjectKind = 'sphere' | 'cylinder' | 'box' | 'none';

export interface SimSettings {
  cableForces: [number, number, number];
  stiffness: number;
  damping: number;
  cableFriction: number;
  bodyFriction: number;
  gravity: number;
  totalMass: number;
  timeScale: number;
  mount: Mount;
  objectKind: ObjectKind;
  objectSize: number;
  objectMass: number;
}

export const DEFAULT_SETTINGS: SimSettings = {
  cableForces: [0, 0, 0],
  stiffness: 0.09,
  damping: 0.0008,
  cableFriction: 0.08,
  bodyFriction: 0.72,
  gravity: 9.81,
  totalMass: 0.0384,
  timeScale: 1,
  mount: 'horizontal',
  objectKind: 'sphere',
  objectSize: 52,
  objectMass: 0.045,
};

export type AutoGraspPhase = 'packing' | 'reaching' | 'wrapping' | 'grasping' | 'holding';

export interface AutoGraspCommand {
  phase: AutoGraspPhase;
  cableForces: [number, number, number];
  wrapProgress: number;
  releaseTarget: boolean;
}

/**
 * Paper-derived antagonistic sequence adapted to three equally spaced cables.
 * One cable is the packing side; the vector sum of the other two acts as the
 * opposing cable in the paper's planar two-cable sequence.
 */
export function autoGraspCommand(elapsedSeconds: number, primaryCable: number): AutoGraspCommand {
  const t = Math.max(0, elapsedSeconds);
  let phase: AutoGraspPhase;
  let packingForce = 0;
  let opposingForce = 0;
  let wrapProgress = 0;

  if (t < 1.2) {
    phase = 'packing';
    packingForce = 6 * (t / 1.2);
  } else if (t < 4.4) {
    phase = 'reaching';
    const progress = (t - 1.2) / 3.2;
    packingForce = 6;
    opposingForce = 5.82 * progress;
  } else if (t < 8.4) {
    phase = 'wrapping';
    const progress = (t - 4.4) / 4;
    // Keep the opposing side steady and relax the original packing side.
    // This is the no-slip surface-climbing step described in Figure 3A.
    packingForce = 6 - 0.8 * progress;
    opposingForce = 5.82;
    wrapProgress = progress;
  } else if (t < 10.2) {
    phase = 'grasping';
    const progress = (t - 8.4) / 1.8;
    packingForce = 5.2;
    opposingForce = 5.82 + 3.18 * progress;
    wrapProgress = 1;
  } else {
    phase = 'holding';
    packingForce = 5.2;
    opposingForce = 9;
    wrapProgress = 1;
  }

  const cableForces: [number, number, number] = [opposingForce, opposingForce, opposingForce];
  cableForces[((primaryCable % 3) + 3) % 3] = packingForce;
  return { phase, cableForces, wrapProgress, releaseTarget: phase === 'holding' };
}

const MM = 0.001;

const CONTACT_STIFFNESS = 220;
const CONTACT_DAMPING = 0.28;
const CONTACT_SLOP = 0.0008;

interface SurfaceSample {
  /** Signed distance to the object enlarged by `margin` (negative inside). */
  distance: number;
  /** Unit normal pointing from the object toward the sample point. */
  normal: CANNON.Vec3;
}

function rodDampingRatio(calibrationValue: number) {
  return Math.min(1.8, Math.max(0.15, calibrationValue * 1000));
}

function mountQuaternion(mount: Mount) {
  const q = new CANNON.Quaternion();
  if (mount === 'hanging') q.setFromAxisAngle(new CANNON.Vec3(0, 0, 1), -Math.PI / 2);
  if (mount === 'upright') q.setFromAxisAngle(new CANNON.Vec3(0, 0, 1), Math.PI / 2);
  return q;
}

export class SpiRobPhysics {
  world = new CANNON.World({ gravity: new CANNON.Vec3(0, -9.81, 0) });
  bodies: CANNON.Body[] = [];
  settings: SimSettings;
  objectBody?: CANNON.Body;
  contactCount = 0;
  objectContactCount = 0;
  graspForce = 0;
  graspQuality = 0;
  isGrasping = false;
  contactingUnits: number[] = [];
  wrapProgress = 0;
  simulatedTime = 0;
  private constraints: CANNON.PointToPointConstraint[] = [];
  private accumulator = 0;
  private bendY = UNIT_DATA.map(() => 0);
  private bendZ = UNIT_DATA.map(() => 0);
  private bendVelocityY = UNIT_DATA.map(() => 0);
  private bendVelocityZ = UNIT_DATA.map(() => 0);
  private unitMasses = UNIT_DATA.map(() => 0);
  private reactionMomentY = UNIT_DATA.map(() => 0);
  private reactionMomentZ = UNIT_DATA.map(() => 0);
  private graspLatched = false;
  private graspReferenceUnit = -1;
  private graspLocalOffset = new CANNON.Vec3();
  private graspBaseOffset = new CANNON.Vec3();
  private wrapDirection = 1;
  private previousWrapProgress = 0;
  private smoothedWrapAnchor: CANNON.Vec3 | null = null;
  private bodyMaterial = new CANNON.Material('TPU 95A');
  private floorMaterial = new CANNON.Material('laboratory surface');
  private objectMaterial = new CANNON.Material('grasped object');
  private initialMount = new CANNON.Quaternion();

  constructor(settings: SimSettings) {
    this.settings = settings;
    this.world.allowSleep = true;
    this.world.broadphase = new CANNON.SAPBroadphase(this.world);
    (this.world.solver as CANNON.GSSolver).iterations = 28;
    (this.world.solver as CANNON.GSSolver).tolerance = 1e-7;
    this.world.defaultContactMaterial.contactEquationStiffness = 2e5;
    this.world.defaultContactMaterial.contactEquationRelaxation = 8;
    this.build();
  }

  private build() {
    this.initialMount = mountQuaternion(this.settings.mount);
    this.world.gravity.set(0, -this.settings.gravity, 0);
    const floor = new CANNON.Body({ mass: 0, material: this.floorMaterial });
    floor.addShape(new CANNON.Plane());
    floor.quaternion.setFromEuler(-Math.PI / 2, 0, 0);
    floor.position.set(0, -0.12, 0);
    floor.collisionFilterGroup = 4;
    floor.collisionFilterMask = 3;
    this.world.addBody(floor);

    const contact = new CANNON.ContactMaterial(this.bodyMaterial, this.floorMaterial, {
      friction: this.settings.bodyFriction,
      restitution: 0.03,
      contactEquationStiffness: 2e5,
      contactEquationRelaxation: 8,
    });
    this.world.addContactMaterial(contact);
    this.world.addContactMaterial(new CANNON.ContactMaterial(this.objectMaterial, this.floorMaterial, {
      friction: 0.92,
      restitution: 0.02,
      contactEquationStiffness: 2e5,
      contactEquationRelaxation: 8,
      frictionEquationStiffness: 8e4,
    }));
    this.world.addContactMaterial(new CANNON.ContactMaterial(this.bodyMaterial, this.objectMaterial, {
      friction: this.settings.bodyFriction,
      restitution: 0.01,
      contactEquationStiffness: 1.4e5,
      contactEquationRelaxation: 10,
    }));

    const root = new CANNON.Vec3(-0.06, 0.08, 0);
    const baseCenter = UNIT_DATA[0][0];
    const volumes = UNIT_DATA.map(([, length, h, w]) => length * h * w);
    const volumeSum = volumes.reduce((a, b) => a + b, 0);

    UNIT_DATA.forEach(([xc, length, h, w], i) => {
      const along = (baseCenter - xc) * MM;
      const localPosition = new CANNON.Vec3(along, 0, 0);
      const worldPosition = this.initialMount.vmult(localPosition);
      worldPosition.vadd(root, worldPosition);
      const physicalMass = this.settings.totalMass * (volumes[i] / volumeSum);
      this.unitMasses[i] = physicalMass;
      const body = new CANNON.Body({
        mass: 0,
        type: i === 0 ? CANNON.Body.STATIC : CANNON.Body.KINEMATIC,
        material: this.bodyMaterial,
        position: worldPosition,
        quaternion: this.initialMount.clone(),
        linearDamping: 0.035,
        angularDamping: 0.08,
        // Kinematic links are driven by the rod solver on every substep.
        // Sleeping them freezes position integration while their commanded
        // velocity keeps changing, so robot links must always stay awake.
        allowSleep: false,
        sleepSpeedLimit: 0.015,
        sleepTimeLimit: 0.7,
      });
      const shape = new CANNON.Box(new CANNON.Vec3(length * MM * 0.47, h * MM * 0.43, w * MM * 0.43));
      body.addShape(shape);
      body.collisionFilterGroup = 1;
      // Robot/object interaction uses the compliant STL-proximity model
      // below. The conservative box shapes still collide with the floor.
      body.collisionFilterMask = 4;
      this.world.addBody(body);
      this.bodies.push(body);
    });

    this.rebuildObject();
  }

  rebuildObject() {
    if (this.objectBody) this.world.removeBody(this.objectBody);
    this.objectBody = undefined;
    if (this.settings.objectKind === 'none') return;
    const radius = this.settings.objectSize * MM * 0.5;
    const body = new CANNON.Body({
      mass: this.settings.objectMass,
      material: this.objectMaterial,
      // Free object pose. AUTO GRASP moves it to the repeatable presentation
      // fixture pose before the contact sequence begins.
      position: new CANNON.Vec3(0.08, -0.12 + radius, 0),
      linearDamping: 0.28,
      angularDamping: 0.36,
    });
    body.collisionFilterGroup = 2;
    body.collisionFilterMask = 4;
    if (this.settings.objectKind === 'sphere') body.addShape(new CANNON.Sphere(radius));
    if (this.settings.objectKind === 'box') body.addShape(new CANNON.Box(new CANNON.Vec3(radius, radius, radius)));
    if (this.settings.objectKind === 'cylinder') {
      const cylinder = new CANNON.Cylinder(radius, radius, radius * 2, 24);
      body.addShape(cylinder);
    }
    this.world.addBody(body);
    this.objectBody = body;
  }

  reset() {
    for (const constraint of this.constraints) this.world.removeConstraint(constraint);
    for (const body of [...this.bodies]) this.world.removeBody(body);
    if (this.objectBody) this.world.removeBody(this.objectBody);
    this.bodies = [];
    this.constraints = [];
    this.objectBody = undefined;
    this.simulatedTime = 0;
    this.accumulator = 0;
    this.contactCount = 0;
    this.objectContactCount = 0;
    this.graspForce = 0;
    this.graspQuality = 0;
    this.isGrasping = false;
    this.contactingUnits = [];
    this.bendY.fill(0);
    this.bendZ.fill(0);
    this.bendVelocityY.fill(0);
    this.bendVelocityZ.fill(0);
    this.reactionMomentY.fill(0);
    this.reactionMomentZ.fill(0);
    this.graspLatched = false;
    this.graspReferenceUnit = -1;
    this.wrapProgress = 0;
    this.previousWrapProgress = 0;
    this.smoothedWrapAnchor = null;
    this.build();
  }

  releaseGrasp() {
    this.graspLatched = false;
    this.graspReferenceUnit = -1;
    this.isGrasping = false;
    this.wrapProgress = 0;
    this.previousWrapProgress = 0;
    this.smoothedWrapAnchor = null;
  }

  /**
   * Signed-distance query for the actual selected test-object shape. Keeping
   * this in one place makes contact, non-penetration and telemetry agree.
   */
  private sampleObjectSurface(worldPoint: CANNON.Vec3, margin = 0): SurfaceSample {
    if (!this.objectBody || this.settings.objectKind === 'none') {
      return { distance: Infinity, normal: new CANNON.Vec3(1, 0, 0) };
    }

    const halfSize = this.settings.objectSize * MM * 0.5;
    const localPoint = this.objectBody.quaternion.inverse().vmult(worldPoint.vsub(this.objectBody.position));
    let distance = Infinity;
    const localNormal = new CANNON.Vec3(1, 0, 0);

    if (this.settings.objectKind === 'sphere') {
      const length = localPoint.length();
      distance = length - (halfSize + margin);
      if (length > 1e-9) localPoint.scale(1 / length, localNormal);
    } else if (this.settings.objectKind === 'box') {
      // Rounded-box SDF: expanding all half extents by the unit radius is a
      // conservative Minkowski approximation for the tapered robot section.
      const extent = halfSize + margin;
      const qx = Math.abs(localPoint.x) - extent;
      const qy = Math.abs(localPoint.y) - extent;
      const qz = Math.abs(localPoint.z) - extent;
      const ox = Math.max(qx, 0);
      const oy = Math.max(qy, 0);
      const oz = Math.max(qz, 0);
      const outside = Math.hypot(ox, oy, oz);
      distance = outside + Math.min(Math.max(qx, qy, qz), 0);
      if (outside > 1e-9) {
        localNormal.set(
          Math.sign(localPoint.x || 1) * ox / outside,
          Math.sign(localPoint.y || 1) * oy / outside,
          Math.sign(localPoint.z || 1) * oz / outside,
        );
      } else if (qx >= qy && qx >= qz) {
        localNormal.set(Math.sign(localPoint.x || 1), 0, 0);
      } else if (qy >= qz) {
        localNormal.set(0, Math.sign(localPoint.y || 1), 0);
      } else {
        localNormal.set(0, 0, Math.sign(localPoint.z || 1));
      }
    } else {
      // Capped cylinder, whose axis is local Y (the same convention used by
      // THREE.CylinderGeometry and cannon-es' Cylinder).
      const radialLength = Math.hypot(localPoint.x, localPoint.z);
      const radialDistance = radialLength - (halfSize + margin);
      const axialDistance = Math.abs(localPoint.y) - (halfSize + margin);
      const outsideRadial = Math.max(radialDistance, 0);
      const outsideAxial = Math.max(axialDistance, 0);
      const outside = Math.hypot(outsideRadial, outsideAxial);
      distance = outside + Math.min(Math.max(radialDistance, axialDistance), 0);
      const radialX = radialLength > 1e-9 ? localPoint.x / radialLength : 1;
      const radialZ = radialLength > 1e-9 ? localPoint.z / radialLength : 0;
      if (outside > 1e-9) {
        localNormal.set(
          radialX * outsideRadial / outside,
          Math.sign(localPoint.y || 1) * outsideAxial / outside,
          radialZ * outsideRadial / outside,
        );
      } else if (radialDistance >= axialDistance) {
        localNormal.set(radialX, 0, radialZ);
      } else {
        localNormal.set(0, Math.sign(localPoint.y || 1), 0);
      }
    }

    return { distance, normal: this.objectBody.quaternion.vmult(localNormal).unit() };
  }

  objectSignedDistance(worldPoint: CANNON.Vec3, margin = 0) {
    return this.sampleObjectSurface(worldPoint, margin).distance;
  }

  private projectToObjectSurface(worldPoint: CANNON.Vec3, margin: number) {
    const sample = this.sampleObjectSurface(worldPoint, margin);
    return worldPoint.vsub(sample.normal.scale(sample.distance));
  }

  /**
   * The wrap anchor must move continuously. Candidate selection can flip
   * between the two symmetric reach solutions and the start unit changes in
   * discrete steps; any anchor jump becomes a (jump / dt) kinematic velocity
   * spike, so the anchor is rate-limited and kept on the enlarged surface.
   */
  private advanceWrapAnchor(target: CANNON.Vec3, margin: number) {
    if (!this.smoothedWrapAnchor) {
      this.smoothedWrapAnchor = target.clone();
      return this.smoothedWrapAnchor.clone();
    }
    const delta = target.vsub(this.smoothedWrapAnchor);
    const distance = delta.length();
    const maxStep = 0.0035;
    if (distance > maxStep) {
      this.smoothedWrapAnchor.vadd(delta.scale(maxStep / distance), this.smoothedWrapAnchor);
      this.smoothedWrapAnchor.copy(this.projectToObjectSurface(this.smoothedWrapAnchor, margin));
    } else {
      this.smoothedWrapAnchor.copy(target);
    }
    return this.smoothedWrapAnchor.clone();
  }

  private connectProximalChain(
    positions: CANNON.Vec3[],
    endUnit: number,
    endPosition: CANNON.Vec3,
    root: CANNON.Vec3,
  ) {
    // Two-ended FABRIK preserves every measured center spacing while keeping
    // the base fixed and the selected contact unit on the object surface.
    for (let iteration = 0; iteration < 24; iteration++) {
      positions[endUnit].copy(endPosition);
      for (let i = endUnit - 1; i >= 0; i--) {
        const restLength = (UNIT_DATA[i][0] - UNIT_DATA[i + 1][0]) * MM;
        const delta = positions[i].vsub(positions[i + 1]);
        const distance = Math.max(1e-8, delta.length());
        positions[i + 1].vadd(delta.scale(restLength / distance), positions[i]);
      }
      positions[0].copy(root);
      for (let i = 1; i <= endUnit; i++) {
        const restLength = (UNIT_DATA[i - 1][0] - UNIT_DATA[i][0]) * MM;
        const delta = positions[i].vsub(positions[i - 1]);
        const distance = Math.max(1e-8, delta.length());
        positions[i - 1].vadd(delta.scale(restLength / distance), positions[i]);
      }
      // A straight FABRIK chain happily cuts through the target when the
      // anchor sits past the horizon, so intermediate units are pushed back
      // onto the enlarged surface inside the same relaxation loop.
      for (let i = 1; i < endUnit; i++) {
        const robotRadius = Math.min(UNIT_DATA[i][2], UNIT_DATA[i][3]) * MM * 0.43;
        const sample = this.sampleObjectSurface(positions[i], robotRadius + 0.00035);
        if (sample.distance >= 0) continue;
        positions[i].vsub(sample.normal.scale(sample.distance), positions[i]);
      }
    }
    const endpointError = endPosition.vsub(positions[endUnit]);
    for (let i = 1; i <= endUnit; i++) positions[i].vadd(endpointError.scale(i / endUnit), positions[i]);
    positions[endUnit].copy(endPosition);
    // The endpoint-error smear above ignores the surface constraint and can
    // push relaxed interior units back inside the object.
    for (let i = 1; i < endUnit; i++) {
      const robotRadius = Math.min(UNIT_DATA[i][2], UNIT_DATA[i][3]) * MM * 0.43;
      const sample = this.sampleObjectSurface(positions[i], robotRadius + 0.00035);
      if (sample.distance < 0) positions[i].vsub(sample.normal.scale(sample.distance), positions[i]);
    }
  }

  private constrainRodAroundObject(positions: CANNON.Vec3[]) {
    if (!this.objectBody || this.settings.objectKind === 'none') return;
    if (this.objectBody.type !== CANNON.Body.KINEMATIC && !this.graspLatched) return;
    const root = positions[0].clone();

    // XPBD-style shape projection: keep the chain inextensible while every
    // unit center stays outside the target surface. Repeating both constraints
    // makes a penetrating free curve settle as a continuous surface-following
    // arc rather than merely registering a contact at the tip.
    for (let iteration = 0; iteration < 18; iteration++) {
      positions[0].copy(root);
      for (let i = 1; i < positions.length; i++) {
        const restLength = (UNIT_DATA[i - 1][0] - UNIT_DATA[i][0]) * MM;
        const delta = positions[i].vsub(positions[i - 1]);
        const distance = Math.max(1e-8, delta.length());
        const error = (distance - restLength) / distance;
        if (i === 1) {
          positions[i].vsub(delta.scale(error), positions[i]);
        } else {
          const correction = delta.scale(error * 0.5);
          positions[i - 1].vadd(correction, positions[i - 1]);
          positions[i].vsub(correction, positions[i]);
        }
      }
      positions[0].copy(root);

      for (let i = 1; i < positions.length; i++) {
        const robotRadius = Math.min(UNIT_DATA[i][2], UNIT_DATA[i][3]) * MM * 0.43;
        const sample = this.sampleObjectSurface(positions[i], robotRadius + 0.00035);
        if (sample.distance >= 0) continue;
        positions[i].vsub(sample.normal.scale(sample.distance), positions[i]);
      }
    }

    if (this.wrapProgress <= 0.02) {
      this.previousWrapProgress = this.wrapProgress;
      this.smoothedWrapAnchor = null;
      return;
    }
    // Contact-path following for the commanded wrap phase. Distal unit centers
    // are placed on successive surface chords. This models the paper's
    // no-slip "climbing" motion instead of teleporting the tip through the
    // target or treating every target as a sphere.
    const startUnit = Math.max(8, 18 - Math.floor(this.wrapProgress * 10));
    const firstRadius = Math.min(UNIT_DATA[startUnit][2], UNIT_DATA[startUnit][3]) * MM * 0.43;
    if (this.settings.objectKind !== 'sphere') {
      let proximalLength = 0;
      for (let i = 1; i <= startUnit; i++) proximalLength += (UNIT_DATA[i - 1][0] - UNIT_DATA[i][0]) * MM;

      // Choose a reachable point on the enlarged real surface. Sampling is
      // deterministic and inexpensive (once per 120-Hz substep, 72 points).
      let wrapAnchor = positions[startUnit].clone();
      let bestScore = Infinity;
      const sampleDistance = this.settings.objectSize * MM * 2.5 + firstRadius;
      for (let sampleIndex = 0; sampleIndex < 72; sampleIndex++) {
        const angle = sampleIndex / 72 * Math.PI * 2;
        const farPoint = this.objectBody.position.vadd(new CANNON.Vec3(
          Math.cos(angle) * sampleDistance,
          Math.sin(angle) * sampleDistance,
          0,
        ));
        const candidate = this.projectToObjectSurface(farPoint, firstRadius + 0.00035);
        const rootDistance = root.distanceTo(candidate);
        if (rootDistance > proximalLength * 1.002) continue;
        const reachScore = Math.abs(rootDistance - proximalLength * 0.94);
        const continuityReference = this.smoothedWrapAnchor ?? positions[startUnit];
        const continuityScore = Math.sqrt(candidate.distanceSquared(continuityReference)) * 0.18;
        const score = reachScore + continuityScore;
        if (score < bestScore) {
          bestScore = score;
          wrapAnchor = candidate;
        }
      }
      wrapAnchor = this.advanceWrapAnchor(wrapAnchor, firstRadius + 0.00035);

      this.connectProximalChain(positions, startUnit, wrapAnchor, root);
      let surfaceNormal = this.sampleObjectSurface(wrapAnchor, firstRadius + 0.00035).normal;
      let positiveTangent = new CANNON.Vec3(-surfaceNormal.y, surfaceNormal.x, 0);
      if (positiveTangent.lengthSquared() < 1e-10) positiveTangent.set(1, 0, 0);
      positiveTangent.normalize();
      const rawDirection = positions[Math.min(startUnit + 1, positions.length - 1)].vsub(positions[startUnit]).unit();
      if (this.previousWrapProgress <= 0.02) this.wrapDirection = positiveTangent.dot(rawDirection) >= 0 ? 1 : -1;
      this.previousWrapProgress = this.wrapProgress;

      for (let i = startUnit + 1; i < positions.length; i++) {
        const restLength = (UNIT_DATA[i - 1][0] - UNIT_DATA[i][0]) * MM;
        const robotRadius = Math.min(UNIT_DATA[i][2], UNIT_DATA[i][3]) * MM * 0.43;
        let stepLength = restLength;
        let projected = positions[i - 1].clone();
        for (let iteration = 0; iteration < 7; iteration++) {
          let tangent = new CANNON.Vec3(-surfaceNormal.y, surfaceNormal.x, 0);
          if (tangent.lengthSquared() < 1e-10) tangent.set(1, 0, 0);
          tangent.normalize();
          tangent.scale(this.wrapDirection * stepLength, tangent);
          const predictor = positions[i - 1].vadd(tangent);
          projected = this.projectToObjectSurface(predictor, robotRadius + 0.00035);
          const chordLength = Math.max(1e-8, positions[i - 1].distanceTo(projected));
          stepLength *= restLength / chordLength;
          surfaceNormal = this.sampleObjectSurface(projected, robotRadius + 0.00035).normal;
        }
        positions[i].copy(projected);
      }
      return;
    }

    const objectRadius = this.settings.objectSize * MM * 0.5;
    const surfaceStartRadius = objectRadius + firstRadius + 0.00035;
    let proximalLength = 0;
    for (let i = 1; i <= startUnit; i++) proximalLength += (UNIT_DATA[i - 1][0] - UNIT_DATA[i][0]) * MM;
    const rootToCenter = this.objectBody.position.vsub(root);
    rootToCenter.z = 0;
    const centerDistance = Math.max(1e-8, rootToCenter.length());
    const minimumReach = Math.abs(centerDistance - surfaceStartRadius) + 0.0002;
    const maximumReach = centerDistance + surfaceStartRadius - 0.0002;
    const endpointDistance = Math.max(minimumReach, Math.min(maximumReach, proximalLength * 0.94));
    const along = (endpointDistance ** 2 - surfaceStartRadius ** 2 + centerDistance ** 2) / (2 * centerDistance);
    const height = Math.sqrt(Math.max(0, endpointDistance ** 2 - along ** 2));
    const centerDirection = rootToCenter.scale(1 / centerDistance);
    const intersectionBase = root.vadd(centerDirection.scale(along));
    const perpendicular = new CANNON.Vec3(-centerDirection.y, centerDirection.x, 0);
    const candidateA = intersectionBase.vadd(perpendicular.scale(height));
    const candidateB = intersectionBase.vsub(perpendicular.scale(height));
    const anchorReference = this.smoothedWrapAnchor ?? positions[startUnit];
    const chosenAnchor = candidateA.distanceSquared(anchorReference) <= candidateB.distanceSquared(anchorReference) ? candidateA : candidateB;
    const wrapAnchor = this.advanceWrapAnchor(chosenAnchor, firstRadius + 0.00035);

    this.connectProximalChain(positions, startUnit, wrapAnchor, root);

    const radial = wrapAnchor.vsub(this.objectBody.position).unit();
    const rawDirection = positions[Math.min(startUnit + 1, positions.length - 1)].vsub(positions[startUnit]).unit();
    const positiveTangent = new CANNON.Vec3(-radial.y, radial.x, 0);
    if (this.previousWrapProgress <= 0.02) this.wrapDirection = positiveTangent.dot(rawDirection) >= 0 ? 1 : -1;
    const directionSign = this.wrapDirection;
    this.previousWrapProgress = this.wrapProgress;
    positions[startUnit].copy(wrapAnchor);

    for (let i = startUnit + 1; i < positions.length; i++) {
      const restLength = (UNIT_DATA[i - 1][0] - UNIT_DATA[i][0]) * MM;
      const robotRadius = Math.min(UNIT_DATA[i][2], UNIT_DATA[i][3]) * MM * 0.43;
      const surfaceRadius = objectRadius + robotRadius + 0.00035;
      const chordAngle = 2 * Math.asin(Math.min(0.98, restLength / (2 * surfaceRadius)));
      const angle = directionSign * chordAngle;
      const cosine = Math.cos(angle);
      const sine = Math.sin(angle);
      const x = radial.x * cosine - radial.y * sine;
      const y = radial.x * sine + radial.y * cosine;
      radial.set(x, y, 0);
      this.objectBody.position.vadd(radial.scale(surfaceRadius), positions[i]);
    }
  }

  private integrateRod(dt: number) {
    const baseRadius = ((UNIT_DATA[0][2] + UNIT_DATA[0][3]) * 0.25) * MM * 0.76;
    const phases = [0, (Math.PI * 2) / 3, (Math.PI * 4) / 3];
    let accumulatedCableTurn = 0;

    for (let i = 1; i < this.bodies.length; i++) {
      const parent = this.bodies[i - 1];
      const radius = ((UNIT_DATA[i][2] + UNIT_DATA[i][3]) * 0.25) * MM * 0.76;
      // Rotational hinge stiffness is EI / segment length. Because the STL
      // units are geometrically similar, EI scales with r^4 and length with r,
      // hence joint stiffness scales with r^3 (not r^4).
      const scale = Math.max(0.012, Math.pow(radius / baseRadius, 3));
      const stiffness = this.settings.stiffness * scale;
      const localMoment = new CANNON.Vec3();

      if (i > 1) accumulatedCableTurn += Math.hypot(this.bendY[i - 1], this.bendZ[i - 1]);

      for (let cable = 0; cable < 3; cable++) {
        // Capstan attenuation depends on cumulative cable turning angle, not
        // on unit index. A straight robot therefore does not lose tension.
        const attenuation = Math.exp(-this.settings.cableFriction * accumulatedCableTurn);
        const force = this.settings.cableForces[cable] * attenuation;
        const y = radius * Math.cos(phases[cable]);
        const z = radius * Math.sin(phases[cable]);
        localMoment.y += z * force;
        localMoment.z -= y * force;
      }

      // Gravity moment of all distal material, resolved in the parent frame.
      let distalMassMoment = 0;
      const parentDistance = UNIT_DATA[0][0] - UNIT_DATA[i - 1][0];
      for (let j = i; j < this.bodies.length; j++) {
        const distance = (UNIT_DATA[0][0] - UNIT_DATA[j][0] - parentDistance) * MM;
        distalMassMoment += this.unitMasses[j] * Math.max(0, distance);
      }
      const localGravity = parent.quaternion.inverse().vmult(new CANNON.Vec3(0, -this.settings.gravity, 0));
      localMoment.y += -distalMassMoment * localGravity.z;
      localMoment.z += distalMassMoment * localGravity.y;

      // Contact reactions are propagated from a touched unit to every
      // upstream elastic joint. This is what lets the arm climb around an
      // object instead of passing through it as a purely kinematic curve.
      localMoment.y += this.reactionMomentY[i];
      localMoment.z += this.reactionMomentZ[i];

      let targetY = localMoment.y / Math.max(1e-6, stiffness);
      let targetZ = localMoment.z / Math.max(1e-6, stiffness);
      const targetMagnitude = Math.hypot(targetY, targetZ);
      // Δθ = 30° in the paper; contact between adjacent units is the hard stop.
      const angularLimit = Math.PI / 6 * 0.96;
      if (targetMagnitude > angularLimit) {
        targetY *= angularLimit / targetMagnitude;
        targetZ *= angularLimit / targetMagnitude;
      }

      const naturalFrequency = 13 + 8 * (i / (this.bodies.length - 1));
      const dampingRatio = rodDampingRatio(this.settings.damping);
      const accelerationY = naturalFrequency ** 2 * (targetY - this.bendY[i]) - 2 * dampingRatio * naturalFrequency * this.bendVelocityY[i];
      const accelerationZ = naturalFrequency ** 2 * (targetZ - this.bendZ[i]) - 2 * dampingRatio * naturalFrequency * this.bendVelocityZ[i];
      this.bendVelocityY[i] += accelerationY * dt;
      this.bendVelocityZ[i] += accelerationZ * dt;
      this.bendY[i] += this.bendVelocityY[i] * dt;
      this.bendZ[i] += this.bendVelocityZ[i] * dt;
      const magnitude = Math.hypot(this.bendY[i], this.bendZ[i]);
      if (magnitude > angularLimit) {
        this.bendY[i] *= angularLimit / magnitude;
        this.bendZ[i] *= angularLimit / magnitude;
        this.bendVelocityY[i] *= 0.28;
        this.bendVelocityZ[i] *= 0.28;
      }
    }

    // Exact forward kinematics followed by object-aware position constraints.
    const targetPositions = this.bodies.map((body) => body.position.clone());
    const targetQuaternions = this.bodies.map((body) => body.quaternion.clone());
    for (let i = 1; i < this.bodies.length; i++) {
      const parentPosition = targetPositions[i - 1];
      const parentQuaternion = targetQuaternions[i - 1];
      const separation = (UNIT_DATA[i - 1][0] - UNIT_DATA[i][0]) * MM;
      const angle = Math.hypot(this.bendY[i], this.bendZ[i]);
      const relative = new CANNON.Quaternion();
      if (angle > 1e-8) relative.setFromAxisAngle(new CANNON.Vec3(0, this.bendY[i] / angle, this.bendZ[i] / angle), angle);
      const targetQuaternion = parentQuaternion.mult(relative);
      const parentLength = UNIT_DATA[i - 1][1] * MM;
      const childLength = UNIT_DATA[i][1] * MM;
      const gap = Math.max(0, separation - (parentLength + childLength) * 0.5);
      const parentHalf = parentQuaternion.vmult(new CANNON.Vec3(parentLength * 0.5 + gap * 0.5, 0, 0));
      const childHalf = targetQuaternion.vmult(new CANNON.Vec3(childLength * 0.5 + gap * 0.5, 0, 0));
      const targetPosition = parentPosition.vadd(parentHalf).vadd(childHalf);
      targetPositions[i].copy(targetPosition);
      targetQuaternions[i].copy(targetQuaternion);
    }

    this.constrainRodAroundObject(targetPositions);

    // Rate-limit how far any unit may travel in one substep. When the wrap
    // constraint engages (or the anchor/start unit shifts) the constrained
    // target can sit far from the body; without a limit that distance turns
    // into a (gap / dt) velocity spike that kicks the solver and the target.
    const followObject = !!this.objectBody
      && this.settings.objectKind !== 'none'
      && (this.objectBody.type === CANNON.Body.KINEMATIC || this.graspLatched);
    const maxTravel = 0.008;
    for (let i = 1; i < this.bodies.length; i++) {
      const body = this.bodies[i];
      const delta = targetPositions[i].vsub(body.position);
      const distance = delta.length();
      if (distance > maxTravel) {
        body.position.vadd(delta.scale(maxTravel / distance), targetPositions[i]);
        // The shortened path may cut into the target object; keep the
        // clamped waypoint outside the enlarged surface as well.
        if (followObject) {
          const robotRadius = Math.min(UNIT_DATA[i][2], UNIT_DATA[i][3]) * MM * 0.43;
          const sample = this.sampleObjectSurface(targetPositions[i], robotRadius + 0.00035);
          if (sample.distance < 0) targetPositions[i].vsub(sample.normal.scale(sample.distance), targetPositions[i]);
        }
      }
    }

    for (let i = 1; i < this.bodies.length; i++) {
      const child = this.bodies[i];
      const intendedDirection = targetQuaternions[i].vmult(new CANNON.Vec3(1, 0, 0)).unit();
      const constrainedDirection = targetPositions[i].vsub(targetPositions[i - 1]).unit();
      const surfaceCorrection = new CANNON.Quaternion();
      surfaceCorrection.setFromVectors(intendedDirection, constrainedDirection);
      const constrainedQuaternion = surfaceCorrection.mult(targetQuaternions[i]);
      child.velocity.set(
        (targetPositions[i].x - child.position.x) / dt,
        (targetPositions[i].y - child.position.y) / dt,
        (targetPositions[i].z - child.position.z) / dt,
      );
      child.wakeUp();
      child.quaternion.copy(constrainedQuaternion);
      child.angularVelocity.setZero();
    }
  }

  private accumulateContactFeedback(dt: number) {
    const nextY = UNIT_DATA.map(() => 0);
    const nextZ = UNIT_DATA.map(() => 0);
    const contactingUnits = new Set<number>();
    const objectNormals: CANNON.Vec3[] = [];
    let normalForce = 0;

    const recordContact = (unitIndex: number, forceOnRobot: CANNON.Vec3, contactPoint: CANNON.Vec3, solvedForce: number) => {
      if (solvedForce <= 1e-6) return;
      contactingUnits.add(unitIndex);
      normalForce += solvedForce;
      objectNormals.push(forceOnRobot.negate().unit());
      for (let joint = 1; joint <= unitIndex; joint++) {
        const parent = this.bodies[joint - 1];
        const separation = (UNIT_DATA[joint - 1][0] - UNIT_DATA[joint][0]) * MM;
        const jointPoint = parent.position.vadd(parent.quaternion.vmult(new CANNON.Vec3(separation * 0.5, 0, 0)));
        const lever = contactPoint.vsub(jointPoint);
        const localMoment = parent.quaternion.inverse().vmult(lever.cross(forceOnRobot));
        nextY[joint] += localMoment.y;
        nextZ[joint] += localMoment.z;
      }
    };

    if (this.objectBody) {
      for (const contact of this.world.contacts) {
        const robotIsBi = this.bodies.includes(contact.bi) && contact.bj === this.objectBody;
        const robotIsBj = this.bodies.includes(contact.bj) && contact.bi === this.objectBody;
        if (!robotIsBi && !robotIsBj) continue;

        const robotBody = robotIsBi ? contact.bi : contact.bj;
        const unitIndex = this.bodies.indexOf(robotBody);
        if (unitIndex < 0) continue;
        // Cannon's normal points bi -> bj. The solved multiplier is a force.
        const solvedForce = Math.min(50, Math.max(0, contact.multiplier || 0));
        const forceOnRobot = contact.ni.scale(robotIsBi ? -solvedForce : solvedForce);
        const contactPoint = robotBody.position.vadd(robotIsBi ? contact.ri : contact.rj);
        recordContact(unitIndex, forceOnRobot, contactPoint, solvedForce);
      }

      // Shape-aware compliant proximity contact closes the small gap between
      // the tapered STL section and the conservative rigid collision proxy.
      for (let unitIndex = 1; unitIndex < this.bodies.length; unitIndex++) {
        if (contactingUnits.has(unitIndex)) continue;
        const robotBody = this.bodies[unitIndex];
        const robotRadius = Math.min(UNIT_DATA[unitIndex][2], UNIT_DATA[unitIndex][3]) * MM * 0.43;
        const surface = this.sampleObjectSurface(robotBody.position, robotRadius);
        const contactSlop = this.objectBody.type === CANNON.Body.KINEMATIC || this.graspLatched ? CONTACT_SLOP : 0;
        const penetration = contactSlop - surface.distance;
        if (penetration <= 0) continue;
        const closingSpeed = Math.max(0, this.objectBody.velocity.vsub(robotBody.velocity).dot(surface.normal));
        // Soft TPU contact: a low penalty stiffness lets floor friction resist
        // the approach while the rod deforms around the target.
        const solvedForce = Math.min(18, Math.max(0, penetration * CONTACT_STIFFNESS + closingSpeed * CONTACT_DAMPING));
        const forceOnRobot = surface.normal.scale(solvedForce);
        const forceOnObject = forceOnRobot.negate();
        const contactPoint = robotBody.position.vsub(surface.normal.scale(robotRadius));
        // Before closure this force moves the free target normally. After
        // closure the distributed grasp constraint already represents the
        // balanced multi-contact resultant, so applying it again would eject
        // the object from a one-sided proxy sample.
        if (!this.graspLatched) this.objectBody.applyForce(forceOnObject, contactPoint);
        recordContact(unitIndex, forceOnRobot, contactPoint, solvedForce);
      }
    }

    const blend = 1 - Math.exp(-dt * 18);
    for (let i = 1; i < this.bodies.length; i++) {
      // Clamp reaction moments to the actuation range and low-pass them to
      // avoid injecting solver jitter into the compliant rod.
      nextY[i] = Math.max(-0.24, Math.min(0.24, nextY[i]));
      nextZ[i] = Math.max(-0.24, Math.min(0.24, nextZ[i]));
      this.reactionMomentY[i] += (nextY[i] - this.reactionMomentY[i]) * blend;
      this.reactionMomentZ[i] += (nextZ[i] - this.reactionMomentZ[i]) * blend;
    }

    let closure = 0;
    for (let a = 0; a < objectNormals.length; a++) {
      for (let b = a + 1; b < objectNormals.length; b++) {
        closure = Math.max(closure, Math.max(0, -objectNormals[a].dot(objectNormals[b])));
      }
    }
    const distinctUnits = contactingUnits.size;
    const weight = Math.max(0.01, this.settings.objectMass * this.settings.gravity);
    const forceScore = Math.min(1, normalForce / (weight * 2.5));
    const coverageScore = Math.min(1, Math.max(0, distinctUnits - 1) / 3);
    const frictionClosure = distinctUnits >= 3
      ? Math.min(1, normalForce * this.settings.bodyFriction / (weight * 1.5)) * coverageScore
      : 0;
    // A soft enveloping gripper may be stable through friction closure even
    // when no pair of contact normals is exactly opposed.
    const rawQuality = Math.max(closure * forceScore * coverageScore, frictionClosure * 0.72);
    this.graspForce += (normalForce - this.graspForce) * blend;
    this.graspQuality += (rawQuality - this.graspQuality) * blend;
    if (!this.graspLatched && this.objectBody && distinctUnits >= 3 && rawQuality >= 0.34) {
      const active = [...contactingUnits].sort((a, b) => a - b);
      this.graspReferenceUnit = active[Math.floor(active.length / 2)];
      this.graspLocalOffset = this.bodies[this.graspReferenceUnit].pointToLocalFrame(this.objectBody.position);
      this.graspBaseOffset = this.bodies[0].pointToLocalFrame(this.objectBody.position);
      this.graspLatched = true;
    }

    const meanCableForce = this.settings.cableForces.reduce((sum, force) => sum + force, 0) / 3;
    if (this.graspLatched) {
      const heldNormalForce = meanCableForce * 0.55;
      this.graspForce += (heldNormalForce - this.graspForce) * blend;
      this.graspQuality += (Math.max(rawQuality, 0.68) - this.graspQuality) * blend;
    }
    if (this.graspLatched && this.objectBody && this.graspReferenceUnit >= 0) {
      const reference = this.bodies[this.graspReferenceUnit];
      // The robot base is fixed in this simulator, so a captured base-frame
      // anchor represents the resultant of the distributed whole-body grasp.
      // A single moving contact frame creates a circular dependency with the
      // surface projection and can numerically eject the object.
      const target = this.bodies[0].pointToWorldFrame(this.graspBaseOffset);
      const error = target.vsub(this.objectBody.position);
      const distance = error.length();
      const targetIsStaged = this.objectBody.type === CANNON.Body.KINEMATIC;
      if (meanCableForce < 1.5 || (!targetIsStaged && distance > this.settings.objectSize * MM * 5)) {
        this.graspLatched = false;
        this.graspReferenceUnit = -1;
      } else if (targetIsStaged) {
        // While the test fixture is holding the target, keep the capture
        // frame current. Dynamic holding begins only after fixture release.
        this.graspLocalOffset = reference.pointToLocalFrame(this.objectBody.position);
        this.graspBaseOffset = this.bodies[0].pointToLocalFrame(this.objectBody.position);
      } else {
        const pretensionNormalForce = meanCableForce * 0.55;
        const maxHoldingForce = this.settings.bodyFriction * Math.max(normalForce, this.graspForce, pretensionNormalForce);
        // Position-based compliant grasp constraint. Limiting correction by
        // F/m·dt² preserves the calibrated holding-force ceiling without the
        // light-object instability of an explicit stiff spring.
        const maxCorrection = maxHoldingForce / Math.max(1e-5, this.settings.objectMass) * dt * dt;
        // The rod re-follows the object surface one substep later, so any
        // single-substep object step reappears as that much penetration.
        // 1 mm per substep (0.12 m/s) is ample for recentring drift.
        const maxObjectStep = 0.001;
        const correctionRatio = distance > 1e-8
          ? Math.min(1 - Math.exp(-dt * 42), maxCorrection / distance, maxObjectStep / distance)
          : 0;
        this.objectBody.position.vadd(error.scale(correctionRatio), this.objectBody.position);
        this.objectBody.velocity.scale(0.32, this.objectBody.velocity);
        this.objectBody.angularVelocity.scale(0.55, this.objectBody.angularVelocity);
        this.objectBody.aabbNeedsUpdate = true;
      }
    }

    this.objectContactCount = distinctUnits;
    this.contactingUnits = [...contactingUnits].sort((a, b) => a - b);
    this.isGrasping = this.graspLatched || (distinctUnits >= 3 && this.graspQuality >= 0.42);
  }

  step(dt: number) {
    this.world.gravity.y = -this.settings.gravity;
    this.contactCount = 0;
    const scaled = Math.min(1 / 20, dt * this.settings.timeScale);
    const fixed = 1 / 120;
    this.accumulator = Math.min(this.accumulator + scaled, fixed * 8);
    let substeps = 0;
    while (this.accumulator >= fixed && substeps < 8) {
      this.integrateRod(fixed);
      this.world.step(fixed);
      this.accumulateContactFeedback(fixed);
      this.accumulator -= fixed;
      this.simulatedTime += fixed;
      substeps++;
    }
    this.contactCount = this.objectContactCount;
  }

  tipSpeed() {
    return this.bodies[this.bodies.length - 1]?.velocity.length() ?? 0;
  }
}

/**
 * An orbit camera for the vanilla pages: drag to orbit, right-drag (or
 * shift-drag) to pan, wheel to zoom. It hands out what a raster pass needs (a
 * view-projection matrix) and what a ray-cast pass needs (origin and the
 * three basis vectors, scaled so `forward + x * right + y * up` is the ray
 * through NDC `(x, y)`), and packs camera rays for `GpuRays`.
 */

import { RAY_FLOATS } from '@pmndrs/bvh';
import { mat4, vec3, type Vec3 } from './math.ts';

export interface CameraBasis {
    readonly origin: Vec3;
    readonly forward: Vec3;
    /** Right, scaled by `tan(fovY / 2) * aspect`. */
    readonly right: Vec3;
    /** Up, scaled by `tan(fovY / 2)`. */
    readonly up: Vec3;
}

export class OrbitCamera {
    target: Vec3 = [0, 0, 0];
    distance = 10;
    yaw = 0.7;
    pitch = 0.45;
    readonly fovY = (50 * Math.PI) / 180;
    near = 0.05;
    far = 500;
    /** Bumped whenever the view changes (pages redraw or re-trace on it). */
    version = 0;

    constructor(private readonly element: HTMLElement) {
        let drag: { x: number; y: number; pan: boolean } | null = null;
        element.addEventListener('contextmenu', (event) => event.preventDefault());
        element.addEventListener('pointerdown', (event) => {
            drag = { x: event.clientX, y: event.clientY, pan: event.button !== 0 || event.shiftKey };
            element.setPointerCapture(event.pointerId);
        });
        element.addEventListener('pointerup', (event) => {
            drag = null;
            element.releasePointerCapture(event.pointerId);
        });
        element.addEventListener('pointermove', (event) => {
            if (!drag) return;
            const dx = event.clientX - drag.x;
            const dy = event.clientY - drag.y;
            drag.x = event.clientX;
            drag.y = event.clientY;
            if (drag.pan) {
                const basis = this.basis(1);
                const scale = (this.distance * 2 * Math.tan(this.fovY / 2)) / element.clientHeight;
                const right = vec3.normalize(basis.right);
                const up = vec3.normalize(basis.up);
                this.target = vec3.add(this.target, vec3.add(vec3.scale(right, -dx * scale), vec3.scale(up, dy * scale)));
            } else {
                this.yaw -= dx * 0.005;
                this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch + dy * 0.005));
            }
            this.version++;
        });
        element.addEventListener(
            'wheel',
            (event) => {
                event.preventDefault();
                this.distance *= Math.exp(event.deltaY * 0.001);
                this.version++;
            },
            { passive: false },
        );
    }

    get position(): Vec3 {
        const c = Math.cos(this.pitch);
        return vec3.add(this.target, vec3.scale([Math.sin(this.yaw) * c, Math.sin(this.pitch), Math.cos(this.yaw) * c], this.distance));
    }

    basis(aspect: number): CameraBasis {
        const origin = this.position;
        const forward = vec3.normalize(vec3.sub(this.target, origin));
        const rightUnit = vec3.normalize(vec3.cross(forward, [0, 1, 0]));
        const upUnit = vec3.cross(rightUnit, forward);
        const tan = Math.tan(this.fovY / 2);
        return { origin, forward, right: vec3.scale(rightUnit, tan * aspect), up: vec3.scale(upUnit, tan) };
    }

    viewProjection(aspect: number): Float32Array {
        return mat4.multiply(mat4.perspective(this.fovY, aspect, this.near, this.far), mat4.lookAt(this.position, this.target, [0, 1, 0]));
    }

    /** The unit direction through NDC `(x, y)` (-1 to 1, y up). */
    direction(x: number, y: number, aspect: number): Vec3 {
        const { forward, right, up } = this.basis(aspect);
        return vec3.normalize(vec3.add(forward, vec3.add(vec3.scale(right, x), vec3.scale(up, y))));
    }

    /** The camera ray through NDC `(x, y)`, packed (`RAY_FLOATS` per ray) at ray `at` of `out`. */
    packRay(x: number, y: number, aspect: number, out = new Float32Array(RAY_FLOATS), at = 0): Float32Array {
        const origin = this.position;
        const direction = this.direction(x, y, aspect);
        out.set([origin[0], origin[1], origin[2], 0, direction[0], direction[1], direction[2], 1e30], at * RAY_FLOATS);
        return out;
    }

    /** Looks at a world box (`[min, max]`) from the current direction. */
    frame(box: ArrayLike<number>): void {
        const center: Vec3 = [(box[0] + box[3]) / 2, (box[1] + box[4]) / 2, (box[2] + box[5]) / 2];
        const radius = Math.max(1e-3, Math.hypot(box[3] - box[0], box[4] - box[1], box[5] - box[2]) / 2);
        this.target = center;
        this.distance = (radius / Math.sin(this.fovY / 2)) * 0.9;
        this.near = Math.max(radius / 1000, 1e-3);
        this.far = this.distance * 10;
        this.version++;
    }
}

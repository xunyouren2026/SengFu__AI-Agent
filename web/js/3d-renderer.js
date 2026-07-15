/**
 * AGI Unified Framework - 3D Renderer Library
 * Three.js wrapper for 3D visualization and rendering
 * @version 1.0.0
 * @author AGI Framework Team
 */

(function(global) {
    'use strict';

    // Check if Three.js is loaded
    if (typeof THREE === 'undefined') {
        console.warn('Three.js is required for 3D Renderer. Please include Three.js library.');
    }

    // 3D Scene Manager
    class Scene3D {
        constructor(container, options = {}) {
            this.container = typeof container === 'string' 
                ? document.querySelector(container) 
                : container;
            
            if (!this.container) {
                throw new Error('3D container not found');
            }
            
            this.options = {
                width: this.container.clientWidth,
                height: this.container.clientHeight,
                backgroundColor: 0x000000,
                antialias: true,
                alpha: true,
                shadows: true,
                camera: {
                    fov: 75,
                    near: 0.1,
                    far: 1000,
                    position: { x: 0, y: 0, z: 5 }
                },
                controls: {
                    enabled: true,
                    enableDamping: true,
                    dampingFactor: 0.05,
                    minDistance: 1,
                    maxDistance: 100
                },
                ...options
            };
            
            this.objects = new Map();
            this.animations = new Map();
            this.raycaster = new THREE.Raycaster();
            this.mouse = new THREE.Vector2();
            
            this.init();
        }
        
        init() {
            this.createScene();
            this.createCamera();
            this.createRenderer();
            this.createLights();
            this.createControls();
            this.bindEvents();
            this.animate();
        }
        
        createScene() {
            this.scene = new THREE.Scene();
            this.scene.background = new THREE.Color(this.options.backgroundColor);
            
            // Fog for depth
            this.scene.fog = new THREE.Fog(this.options.backgroundColor, 10, 50);
        }
        
        createCamera() {
            const { fov, near, far, position } = this.options.camera;
            const aspect = this.options.width / this.options.height;
            
            this.camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
            this.camera.position.set(position.x, position.y, position.z);
        }
        
        createRenderer() {
            this.renderer = new THREE.WebGLRenderer({
                antialias: this.options.antialias,
                alpha: this.options.alpha
            });
            
            this.renderer.setSize(this.options.width, this.options.height);
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            
            if (this.options.shadows) {
                this.renderer.shadowMap.enabled = true;
                this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            }
            
            this.container.appendChild(this.renderer.domElement);
        }
        
        createLights() {
            // Ambient light
            this.ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
            this.scene.add(this.ambientLight);
            
            // Directional light
            this.directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            this.directionalLight.position.set(5, 10, 7);
            this.directionalLight.castShadow = true;
            this.directionalLight.shadow.mapSize.width = 2048;
            this.directionalLight.shadow.mapSize.height = 2048;
            this.scene.add(this.directionalLight);
            
            // Point light
            this.pointLight = new THREE.PointLight(0xffffff, 0.5);
            this.pointLight.position.set(-5, 5, -5);
            this.scene.add(this.pointLight);
        }
        
        createControls() {
            if (!this.options.controls.enabled || typeof THREE.OrbitControls === 'undefined') {
                return;
            }
            
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            Object.assign(this.controls, this.options.controls);
        }
        
        bindEvents() {
            // Resize
            window.addEventListener('resize', () => this.handleResize());
            
            // Mouse events for raycasting
            this.renderer.domElement.addEventListener('mousemove', (e) => this.handleMouseMove(e));
            this.renderer.domElement.addEventListener('click', (e) => this.handleClick(e));
        }
        
        handleResize() {
            this.options.width = this.container.clientWidth;
            this.options.height = this.container.clientHeight;
            
            this.camera.aspect = this.options.width / this.options.height;
            this.camera.updateProjectionMatrix();
            
            this.renderer.setSize(this.options.width, this.options.height);
        }
        
        handleMouseMove(event) {
            const rect = this.renderer.domElement.getBoundingClientRect();
            this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        }
        
        handleClick(event) {
            this.raycaster.setFromCamera(this.mouse, this.camera);
            const intersects = this.raycaster.intersectObjects(this.scene.children, true);
            
            if (intersects.length > 0) {
                const object = intersects[0].object;
                if (object.userData.onClick) {
                    object.userData.onClick(object, intersects[0]);
                }
            }
        }
        
        animate() {
            requestAnimationFrame(() => this.animate());
            
            if (this.controls) {
                this.controls.update();
            }
            
            // Run animations
            this.animations.forEach((animation, id) => {
                if (animation.active) {
                    animation.callback(animation.object, animation);
                }
            });
            
            this.renderer.render(this.scene, this.camera);
        }
        
        // Object creation methods
        createMesh(geometry, material, options = {}) {
            const mesh = new THREE.Mesh(geometry, material);
            
            if (options.position) {
                mesh.position.set(options.position.x || 0, options.position.y || 0, options.position.z || 0);
            }
            
            if (options.rotation) {
                mesh.rotation.set(options.rotation.x || 0, options.rotation.y || 0, options.rotation.z || 0);
            }
            
            if (options.scale) {
                mesh.scale.set(options.scale.x || 1, options.scale.y || 1, options.scale.z || 1);
            }
            
            if (options.castShadow !== false) {
                mesh.castShadow = true;
            }
            
            if (options.receiveShadow !== false) {
                mesh.receiveShadow = true;
            }
            
            if (options.userData) {
                mesh.userData = options.userData;
            }
            
            if (options.name) {
                mesh.name = options.name;
                this.objects.set(options.name, mesh);
            }
            
            this.scene.add(mesh);
            return mesh;
        }
        
        createBox(width, height, depth, material, options = {}) {
            const geometry = new THREE.BoxGeometry(width, height, depth);
            return this.createMesh(geometry, material, options);
        }
        
        createSphere(radius, material, options = {}) {
            const segments = options.segments || { width: 32, height: 16 };
            const geometry = new THREE.SphereGeometry(radius, segments.width, segments.height);
            return this.createMesh(geometry, material, options);
        }
        
        createCylinder(radiusTop, radiusBottom, height, material, options = {}) {
            const radialSegments = options.radialSegments || 32;
            const geometry = new THREE.CylinderGeometry(radiusTop, radiusBottom, height, radialSegments);
            return this.createMesh(geometry, material, options);
        }
        
        createPlane(width, height, material, options = {}) {
            const geometry = new THREE.PlaneGeometry(width, height);
            return this.createMesh(geometry, material, options);
        }
        
        createTorus(radius, tube, material, options = {}) {
            const radialSegments = options.radialSegments || 16;
            const tubularSegments = options.tubularSegments || 100;
            const geometry = new THREE.TorusGeometry(radius, tube, radialSegments, tubularSegments);
            return this.createMesh(geometry, material, options);
        }
        
        createLine(points, material, options = {}) {
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geometry, material);
            
            if (options.name) {
                line.name = options.name;
                this.objects.set(options.name, line);
            }
            
            this.scene.add(line);
            return line;
        }
        
        createPoints(positions, material, options = {}) {
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            
            const points = new THREE.Points(geometry, material);
            
            if (options.name) {
                points.name = options.name;
                this.objects.set(options.name, points);
            }
            
            this.scene.add(points);
            return points;
        }
        
        // Material creation
        createMaterial(type, options = {}) {
            const materials = {
                basic: THREE.MeshBasicMaterial,
                lambert: THREE.MeshLambertMaterial,
                phong: THREE.MeshPhongMaterial,
                standard: THREE.MeshStandardMaterial,
                physical: THREE.MeshPhysicalMaterial,
                depth: THREE.MeshDepthMaterial,
                normal: THREE.MeshNormalMaterial,
                line: THREE.LineBasicMaterial,
                points: THREE.PointsMaterial
            };
            
            const MaterialClass = materials[type] || THREE.MeshStandardMaterial;
            return new MaterialClass(options);
        }
        
        // Animation methods
        addAnimation(id, object, callback) {
            this.animations.set(id, {
                id,
                object,
                callback,
                active: true,
                startTime: Date.now(),
                elapsed: 0
            });
        }
        
        removeAnimation(id) {
            this.animations.delete(id);
        }
        
        animateTo(object, target, duration = 1000, easing = 'easeOut') {
            const start = {
                position: object.position.clone(),
                rotation: object.rotation.clone(),
                scale: object.scale.clone()
            };
            
            const startTime = Date.now();
            
            this.addAnimation(`animate_${object.uuid}`, object, (obj, anim) => {
                const elapsed = Date.now() - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const eased = this.ease(progress, easing);
                
                if (target.position) {
                    obj.position.lerpVectors(start.position, new THREE.Vector3(
                        target.position.x || start.position.x,
                        target.position.y || start.position.y,
                        target.position.z || start.position.z
                    ), eased);
                }
                
                if (target.rotation) {
                    obj.rotation.x = start.rotation.x + (target.rotation.x - start.rotation.x) * eased;
                    obj.rotation.y = start.rotation.y + (target.rotation.y - start.rotation.y) * eased;
                    obj.rotation.z = start.rotation.z + (target.rotation.z - start.rotation.z) * eased;
                }
                
                if (target.scale) {
                    obj.scale.lerpVectors(start.scale, new THREE.Vector3(
                        target.scale.x || start.scale.x,
                        target.scale.y || start.scale.y,
                        target.scale.z || start.scale.z
                    ), eased);
                }
                
                if (progress >= 1) {
                    anim.active = false;
                    this.removeAnimation(anim.id);
                }
            });
        }
        
        ease(t, type) {
            switch (type) {
                case 'linear': return t;
                case 'easeIn': return t * t;
                case 'easeOut': return 1 - (1 - t) * (1 - t);
                case 'easeInOut': return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
                default: return t;
            }
        }
        
        rotate(object, axis, speed) {
            this.addAnimation(`rotate_${object.uuid}`, object, (obj) => {
                obj.rotation[axis] += speed;
            });
        }
        
        float(object, amplitude = 0.5, speed = 1) {
            const startY = object.position.y;
            
            this.addAnimation(`float_${object.uuid}`, object, (obj, anim) => {
                anim.elapsed += 0.016;
                obj.position.y = startY + Math.sin(anim.elapsed * speed) * amplitude;
            });
        }
        
        pulse(object, minScale = 0.9, maxScale = 1.1, speed = 2) {
            this.addAnimation(`pulse_${object.uuid}`, object, (obj, anim) => {
                anim.elapsed += 0.016;
                const scale = minScale + (Math.sin(anim.elapsed * speed) + 1) / 2 * (maxScale - minScale);
                obj.scale.setScalar(scale);
            });
        }
        
        // Object management
        getObject(name) {
            return this.objects.get(name);
        }
        
        removeObject(name) {
            const object = this.objects.get(name);
            if (object) {
                this.scene.remove(object);
                
                // Dispose geometry and materials
                if (object.geometry) object.geometry.dispose();
                if (object.material) {
                    if (Array.isArray(object.material)) {
                        object.material.forEach(m => m.dispose());
                    } else {
                        object.material.dispose();
                    }
                }
                
                this.objects.delete(name);
            }
        }
        
        clear() {
            this.objects.forEach((object, name) => {
                this.removeObject(name);
            });
            this.objects.clear();
            this.animations.clear();
        }
        
        // Screenshot
        screenshot(format = 'image/png', quality = 1) {
            this.renderer.render(this.scene, this.camera);
            return this.renderer.domElement.toDataURL(format, quality);
        }
        
        // Export
        exportGLTF(filename = 'scene.gltf') {
            if (typeof THREE.GLTFExporter === 'undefined') {
                console.warn('GLTFExporter not loaded');
                return;
            }
            
            const exporter = new THREE.GLTFExporter();
            exporter.parse(this.scene, (gltf) => {
                const output = JSON.stringify(gltf, null, 2);
                const blob = new Blob([output], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                
                const link = document.createElement('a');
                link.href = url;
                link.download = filename;
                link.click();
                
                URL.revokeObjectURL(url);
            });
        }
        
        // Cleanup
        destroy() {
            this.clear();
            
            if (this.controls) {
                this.controls.dispose();
            }
            
            this.renderer.dispose();
            this.container.removeChild(this.renderer.domElement);
        }
    }

    // Molecule Visualizer
    class MoleculeVisualizer extends Scene3D {
        constructor(container, options = {}) {
            super(container, {
                camera: { position: { x: 0, y: 0, z: 20 } },
                ...options
            });
            
            this.atoms = [];
            this.bonds = [];
            
            // Atom colors (CPK coloring)
            this.atomColors = {
                H: 0xFFFFFF,  // Hydrogen - White
                C: 0x909090,  // Carbon - Gray
                N: 0x3050F8,  // Nitrogen - Blue
                O: 0xFF0D0D,  // Oxygen - Red
                F: 0x90E050,  // Fluorine - Green
                P: 0xFF8000,  // Phosphorus - Orange
                S: 0xFFFF30,  // Sulfur - Yellow
                Cl: 0x1FF01F, // Chlorine - Green
                Fe: 0xE06633, // Iron - Orange-red
                Cu: 0xC78033, // Copper - Brown
                Zn: 0x7D80B0, // Zinc - Gray-blue
                default: 0xFF69B4
            };
            
            // Atom radii (van der Waals radii in Angstroms, scaled)
            this.atomRadii = {
                H: 0.3,
                C: 0.5,
                N: 0.5,
                O: 0.48,
                F: 0.47,
                P: 0.7,
                S: 0.7,
                Cl: 0.7,
                Fe: 0.8,
                Cu: 0.8,
                Zn: 0.8,
                default: 0.6
            };
        }
        
        loadMolecule(moleculeData) {
            this.clearMolecule();
            
            if (moleculeData.atoms) {
                moleculeData.atoms.forEach(atom => this.addAtom(atom));
            }
            
            if (moleculeData.bonds) {
                moleculeData.bonds.forEach(bond => this.addBond(bond));
            }
            
            this.centerCamera();
        }
        
        addAtom(atom) {
            const element = atom.element || 'C';
            const color = this.atomColors[element] || this.atomColors.default;
            const radius = this.atomRadii[element] || this.atomRadii.default;
            
            const material = this.createMaterial('physical', {
                color: color,
                metalness: 0.2,
                roughness: 0.3,
                clearcoat: 0.5
            });
            
            const sphere = this.createSphere(radius, material, {
                position: atom.position,
                name: `atom_${atom.id}`,
                userData: { type: 'atom', ...atom }
            });
            
            this.atoms.push(sphere);
            return sphere;
        }
        
        addBond(bond) {
            const atom1 = this.atoms.find(a => a.userData.id === bond.atom1);
            const atom2 = this.atoms.find(a => a.userData.id === bond.atom2);
            
            if (!atom1 || !atom2) return;
            
            const start = atom1.position;
            const end = atom2.position;
            const distance = start.distanceTo(end);
            
            const material = this.createMaterial('standard', {
                color: 0x666666,
                roughness: 0.5
            });
            
            // Create cylinder for bond
            const cylinder = this.createCylinder(0.1, 0.1, distance, material, {
                name: `bond_${bond.atom1}_${bond.atom2}`,
                userData: { type: 'bond', ...bond }
            });
            
            // Position and orient cylinder
            const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
            cylinder.position.copy(midpoint);
            cylinder.lookAt(end);
            cylinder.rotateX(Math.PI / 2);
            
            this.bonds.push(cylinder);
            return cylinder;
        }
        
        clearMolecule() {
            this.atoms.forEach(atom => {
                this.scene.remove(atom);
                atom.geometry.dispose();
                atom.material.dispose();
            });
            
            this.bonds.forEach(bond => {
                this.scene.remove(bond);
                bond.geometry.dispose();
                bond.material.dispose();
            });
            
            this.atoms = [];
            this.bonds = [];
        }
        
        centerCamera() {
            if (this.atoms.length === 0) return;
            
            const box = new THREE.Box3();
            this.atoms.forEach(atom => box.expandByObject(atom));
            
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            
            this.camera.position.set(center.x, center.y, center.z + maxDim * 2);
            this.controls.target.copy(center);
            this.controls.update();
        }
        
        highlightAtom(atomId, color = 0xFFFF00) {
            const atom = this.atoms.find(a => a.userData.id === atomId);
            if (atom) {
                atom.material.emissive.setHex(color);
                atom.material.emissiveIntensity = 0.5;
            }
        }
        
        clearHighlight() {
            this.atoms.forEach(atom => {
                atom.material.emissive.setHex(0x000000);
                atom.material.emissiveIntensity = 0;
            });
        }
    }

    // Point Cloud Visualizer
    class PointCloudVisualizer extends Scene3D {
        constructor(container, options = {}) {
            super(container, {
                camera: { position: { x: 0, y: 5, z: 10 } },
                ...options
            });
            
            this.pointCloud = null;
        }
        
        loadPoints(points, colors = null) {
            if (this.pointCloud) {
                this.scene.remove(this.pointCloud);
                this.pointCloud.geometry.dispose();
                this.pointCloud.material.dispose();
            }
            
            const geometry = new THREE.BufferGeometry();
            const positions = new Float32Array(points.length * 3);
            
            points.forEach((point, i) => {
                positions[i * 3] = point[0];
                positions[i * 3 + 1] = point[1];
                positions[i * 3 + 2] = point[2];
            });
            
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            
            if (colors) {
                const colorArray = new Float32Array(colors.length * 3);
                colors.forEach((color, i) => {
                    colorArray[i * 3] = color[0];
                    colorArray[i * 3 + 1] = color[1];
                    colorArray[i * 3 + 2] = color[2];
                });
                geometry.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));
            }
            
            const material = this.createMaterial('points', {
                size: this.options.pointSize || 0.05,
                vertexColors: colors ? true : false,
                color: 0xffffff
            });
            
            this.pointCloud = new THREE.Points(geometry, material);
            this.pointCloud.name = 'pointCloud';
            this.scene.add(this.pointCloud);
            
            this.centerOnPoints(points);
        }
        
        centerOnPoints(points) {
            if (points.length === 0) return;
            
            const center = points.reduce((acc, p) => ({
                x: acc.x + p[0] / points.length,
                y: acc.y + p[1] / points.length,
                z: acc.z + p[2] / points.length
            }), { x: 0, y: 0, z: 0 });
            
            this.controls.target.set(center.x, center.y, center.z);
            this.camera.position.set(center.x, center.y + 5, center.z + 10);
            this.controls.update();
        }
    }

    // Neural Network Visualizer
    class NeuralNetworkVisualizer extends Scene3D {
        constructor(container, options = {}) {
            super(container, {
                camera: { position: { x: 0, y: 0, z: 30 } },
                ...options
            });

            this.layers = [];
            this.connections = [];
            this.neuronMeshes = [];
        }

        loadNetwork(architecture) {
            this.clearNetwork();

            const layerSpacing = 8;
            const neuronSpacing = 3;

            // Create neurons for each layer
            architecture.forEach((layerSize, layerIndex) => {
                const layer = [];
                const x = (layerIndex - (architecture.length - 1) / 2) * layerSpacing;

                for (let i = 0; i < layerSize; i++) {
                    const y = (i - (layerSize - 1) / 2) * neuronSpacing;

                    const material = this.createMaterial('physical', {
                        color: 0x4a90e2,
                        metalness: 0.3,
                        roughness: 0.4
                    });

                    const neuron = this.createSphere(0.4, material, {
                        position: { x, y, z: 0 },
                        name: `neuron_${layerIndex}_${i}`
                    });

                    layer.push({ mesh: neuron, x, y, z: 0, index: i });
                }

                this.layers.push(layer);
            });

            // Create connections between layers
            for (let l = 0; l < this.layers.length - 1; l++) {
                const currentLayer = this.layers[l];
                const nextLayer = this.layers[l + 1];

                currentLayer.forEach(neuron1 => {
                    nextLayer.forEach(neuron2 => {
                        this.createConnection(neuron1, neuron2);
                    });
                });
            }

            // Animate activation
            this.animateNetwork();
        }

        createConnection(neuron1, neuron2) {
            const points = [
                new THREE.Vector3(neuron1.x, neuron1.y, neuron1.z),
                new THREE.Vector3(neuron2.x, neuron2.y, neuron2.z)
            ];

            const material = new THREE.LineBasicMaterial({
                color: 0x666666,
                transparent: true,
                opacity: 0.2
            });

            const line = this.createLine(points, material);
            this.connections.push({ line, neuron1, neuron2, active: false });
        }

        animateNetwork() {
            let time = 0;

            this.addAnimation('networkPulse', null, () => {
                time += 0.02;

                // Animate neurons
                this.layers.forEach((layer, layerIndex) => {
                    layer.forEach((neuron, neuronIndex) => {
                        const offset = layerIndex * 0.5 + neuronIndex * 0.1;
                        const scale = 1 + Math.sin(time + offset) * 0.2;
                        neuron.mesh.scale.setScalar(scale);

                        // Change color based on activation
                        const activation = (Math.sin(time + offset) + 1) / 2;
                        const color = new THREE.Color().lerpColors(
                            new THREE.Color(0x4a90e2),
                            new THREE.Color(0xff6b6b),
                            activation
                        );
                        neuron.mesh.material.color = color;
                    });
                });

                // Animate connections
                this.connections.forEach(conn => {
                    if (Math.random() > 0.99) {
                        conn.active = true;
                        conn.line.material.opacity = 0.8;
                        conn.line.material.color = new THREE.Color(0x4a90e2);
                    } else if (conn.active) {
                        conn.line.material.opacity *= 0.95;
                        if (conn.line.material.opacity < 0.2) {
                            conn.active = false;
                            conn.line.material.color = new THREE.Color(0x666666);
                        }
                    }
                });
            });
        }

        clearNetwork() {
            this.layers.forEach(layer => {
                layer.forEach(neuron => {
                    this.scene.remove(neuron.mesh);
                    neuron.mesh.geometry.dispose();
                    neuron.mesh.material.dispose();
                });
            });

            this.connections.forEach(conn => {
                this.scene.remove(conn.line);
                conn.line.geometry.dispose();
                conn.line.material.dispose();
            });

            this.layers = [];
            this.connections = [];
            this.removeAnimation('networkPulse');
        }

        highlightLayer(layerIndex) {
            this.layers.forEach((layer, index) => {
                layer.forEach(neuron => {
                    if (index === layerIndex) {
                        neuron.mesh.material.emissive.setHex(0x444444);
                    } else {
                        neuron.mesh.material.emissive.setHex(0x000000);
                    }
                });
            });
        }
    }

    // Particle System Visualizer
    class ParticleSystemVisualizer extends Scene3D {
        constructor(container, options = {}) {
            super(container, {
                camera: { position: { x: 0, y: 0, z: 50 } },
                ...options
            });

            this.particleCount = options.particleCount || 1000;
            this.particles = null;
            this.particleData = [];
        }

        createParticles(pattern = 'sphere') {
            if (this.particles) {
                this.scene.remove(this.particles);
                this.particles.geometry.dispose();
                this.particles.material.dispose();
            }

            const geometry = new THREE.BufferGeometry();
            const positions = new Float32Array(this.particleCount * 3);
            const colors = new Float32Array(this.particleCount * 3);
            const sizes = new Float32Array(this.particleCount);

            this.particleData = [];

            for (let i = 0; i < this.particleCount; i++) {
                const i3 = i * 3;
                let x, y, z;

                switch (pattern) {
                    case 'sphere':
                        const theta = Math.random() * Math.PI * 2;
                        const phi = Math.acos(2 * Math.random() - 1);
                        const r = 10 + Math.random() * 10;
                        x = r * Math.sin(phi) * Math.cos(theta);
                        y = r * Math.sin(phi) * Math.sin(theta);
                        z = r * Math.cos(phi);
                        break;
                    case 'cube':
                        x = (Math.random() - 0.5) * 30;
                        y = (Math.random() - 0.5) * 30;
                        z = (Math.random() - 0.5) * 30;
                        break;
                    case 'helix':
                        const t = (i / this.particleCount) * Math.PI * 4;
                        const radius = 5 + i / this.particleCount * 10;
                        x = Math.cos(t) * radius;
                        y = (i / this.particleCount - 0.5) * 30;
                        z = Math.sin(t) * radius;
                        break;
                    default:
                        x = (Math.random() - 0.5) * 20;
                        y = (Math.random() - 0.5) * 20;
                        z = (Math.random() - 0.5) * 20;
                }

                positions[i3] = x;
                positions[i3 + 1] = y;
                positions[i3 + 2] = z;

                const color = new THREE.Color();
                color.setHSL(i / this.particleCount, 0.7, 0.5);
                colors[i3] = color.r;
                colors[i3 + 1] = color.g;
                colors[i3 + 2] = color.b;

                sizes[i] = Math.random() * 2 + 0.5;

                this.particleData.push({
                    x, y, z,
                    vx: (Math.random() - 0.5) * 0.1,
                    vy: (Math.random() - 0.5) * 0.1,
                    vz: (Math.random() - 0.5) * 0.1,
                    originalX: x,
                    originalY: y,
                    originalZ: z
                });
            }

            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

            const material = new THREE.PointsMaterial({
                size: 0.5,
                vertexColors: true,
                transparent: true,
                opacity: 0.8,
                sizeAttenuation: true
            });

            this.particles = new THREE.Points(geometry, material);
            this.particles.name = 'particles';
            this.scene.add(this.particles);

            this.animateParticles();
        }

        animateParticles() {
            this.addAnimation('particleMotion', this.particles, (obj, anim) => {
                const positions = obj.geometry.attributes.position.array;

                for (let i = 0; i < this.particleCount; i++) {
                    const i3 = i * 3;
                    const data = this.particleData[i];

                    // Update position
                    data.x += data.vx;
                    data.y += data.vy;
                    data.z += data.vz;

                    // Boundary check
                    const dist = Math.sqrt(data.x ** 2 + data.y ** 2 + data.z ** 2);
                    if (dist > 25) {
                        data.vx *= -1;
                        data.vy *= -1;
                        data.vz *= -1;
                    }

                    positions[i3] = data.x;
                    positions[i3 + 1] = data.y;
                    positions[i3 + 2] = data.z;
                }

                obj.geometry.attributes.position.needsUpdate = true;
            });
        }

        explode() {
            this.particleData.forEach(data => {
                data.vx = (Math.random() - 0.5) * 2;
                data.vy = (Math.random() - 0.5) * 2;
                data.vz = (Math.random() - 0.5) * 2;
            });

            setTimeout(() => {
                this.particleData.forEach(data => {
                    data.vx = (Math.random() - 0.5) * 0.1;
                    data.vy = (Math.random() - 0.5) * 0.1;
                    data.vz = (Math.random() - 0.5) * 0.1;
                });
            }, 1000);
        }

        implode() {
            this.particleData.forEach(data => {
                const dx = data.originalX - data.x;
                const dy = data.originalY - data.y;
                const dz = data.originalZ - data.z;
                data.vx = dx * 0.05;
                data.vy = dy * 0.05;
                data.vz = dz * 0.05;
            });
        }
    }

    // Volume Rendering System
    class VolumeRenderer extends Scene3D {
        constructor(container, options = {}) {
            super(container, {
                camera: { position: { x: 0, y: 0, z: 5 } },
                ...options
            });

            this.volumeData = null;
            this.transferFunction = [];
            this.slicePlanes = [];
        }

        loadVolume(data, dimensions) {
            this.volumeData = {
                data: new Float32Array(data),
                dimensions: dimensions // { x, y, z }
            };

            this.createVolumeTexture();
            this.setupTransferFunction();
            this.createVolumeMesh();
        }

        createVolumeTexture() {
            const { data, dimensions } = this.volumeData;
            
            this.texture = new THREE.Data3DTexture(
                data,
                dimensions.x,
                dimensions.y,
                dimensions.z
            );
            this.texture.format = THREE.RedFormat;
            this.texture.type = THREE.FloatType;
            this.texture.minFilter = THREE.LinearFilter;
            this.texture.magFilter = THREE.LinearFilter;
            this.texture.needsUpdate = true;
        }

        setupTransferFunction() {
            // Default transfer function (opacity and color mapping)
            this.transferFunction = [];
            const steps = 256;
            
            for (let i = 0; i < steps; i++) {
                const t = i / (steps - 1);
                this.transferFunction.push({
                    value: t,
                    color: new THREE.Color().setHSL(0.7 - t * 0.5, 0.8, 0.5),
                    opacity: Math.pow(t, 2) * 0.5
                });
            }

            this.createTransferTexture();
        }

        createTransferTexture() {
            const data = new Uint8Array(256 * 4);
            
            this.transferFunction.forEach((tf, i) => {
                const idx = i * 4;
                data[idx] = Math.floor(tf.color.r * 255);
                data[idx + 1] = Math.floor(tf.color.g * 255);
                data[idx + 2] = Math.floor(tf.color.b * 255);
                data[idx + 3] = Math.floor(tf.opacity * 255);
            });

            this.transferTexture = new THREE.DataTexture(data, 256, 1);
            this.transferTexture.format = THREE.RGBAFormat;
            this.transferTexture.needsUpdate = true;
        }

        createVolumeMesh() {
            // Create bounding box geometry
            const geometry = new THREE.BoxGeometry(2, 2, 2);
            
            // Volume rendering shader material
            const material = new THREE.ShaderMaterial({
                uniforms: {
                    volumeTexture: { value: this.texture },
                    transferTexture: { value: this.transferTexture },
                    steps: { value: 128 },
                    boxMin: { value: new THREE.Vector3(-1, -1, -1) },
                    boxMax: { value: new THREE.Vector3(1, 1, 1) }
                },
                vertexShader: `
                    varying vec3 vPosition;
                    varying vec3 vOrigin;
                    varying vec3 vDirection;
                    
                    void main() {
                        vPosition = position;
                        vOrigin = (modelMatrix * vec4(position, 1.0)).xyz;
                        vDirection = vOrigin - cameraPosition;
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                `,
                fragmentShader: `
                    uniform sampler3D volumeTexture;
                    uniform sampler2D transferTexture;
                    uniform float steps;
                    uniform vec3 boxMin;
                    uniform vec3 boxMax;
                    
                    varying vec3 vPosition;
                    varying vec3 vOrigin;
                    varying vec3 vDirection;
                    
                    vec2 intersectBox(vec3 origin, vec3 dir, vec3 boxMin, vec3 boxMax) {
                        vec3 invDir = 1.0 / dir;
                        vec3 t0 = (boxMin - origin) * invDir;
                        vec3 t1 = (boxMax - origin) * invDir;
                        vec3 tmin = min(t0, t1);
                        vec3 tmax = max(t0, t1);
                        float tNear = max(max(tmin.x, tmin.y), tmin.z);
                        float tFar = min(min(tmax.x, tmax.y), tmax.z);
                        return vec2(tNear, tFar);
                    }
                    
                    void main() {
                        vec3 dir = normalize(vDirection);
                        vec2 bounds = intersectBox(vOrigin, dir, boxMin, boxMax);
                        
                        if (bounds.x > bounds.y) discard;
                        
                        bounds.x = max(bounds.x, 0.0);
                        
                        vec4 color = vec4(0.0);
                        float stepSize = (bounds.y - bounds.x) / steps;
                        
                        for (float t = bounds.x; t < bounds.y; t += stepSize) {
                            vec3 pos = vOrigin + t * dir;
                            vec3 uvw = (pos - boxMin) / (boxMax - boxMin);
                            
                            float intensity = texture(volumeTexture, uvw).r;
                            vec4 tfColor = texture(transferTexture, vec2(intensity, 0.5));
                            
                            color.rgb += (1.0 - color.a) * tfColor.rgb * tfColor.a;
                            color.a += (1.0 - color.a) * tfColor.a;
                            
                            if (color.a > 0.99) break;
                        }
                        
                        gl_FragColor = color;
                    }
                `,
                transparent: true,
                side: THREE.BackSide
            });

            this.volumeMesh = new THREE.Mesh(geometry, material);
            this.scene.add(this.volumeMesh);
        }

        addSlicePlane(axis, position) {
            const geometry = new THREE.PlaneGeometry(2, 2);
            const material = new THREE.ShaderMaterial({
                uniforms: {
                    volumeTexture: { value: this.texture },
                    transferTexture: { value: this.transferTexture },
                    slicePosition: { value: position },
                    sliceAxis: { value: new THREE.Vector3(
                        axis === 'x' ? 1 : 0,
                        axis === 'y' ? 1 : 0,
                        axis === 'z' ? 1 : 0
                    )}
                },
                vertexShader: `
                    varying vec2 vUv;
                    void main() {
                        vUv = uv;
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                `,
                fragmentShader: `
                    uniform sampler3D volumeTexture;
                    uniform sampler2D transferTexture;
                    uniform float slicePosition;
                    uniform vec3 sliceAxis;
                    varying vec2 vUv;
                    
                    void main() {
                        vec3 uvw;
                        if (sliceAxis.x > 0.5) {
                            uvw = vec3(slicePosition, vUv);
                        } else if (sliceAxis.y > 0.5) {
                            uvw = vec3(vUv.x, slicePosition, vUv.y);
                        } else {
                            uvw = vec3(vUv, slicePosition);
                        }
                        
                        float intensity = texture(volumeTexture, uvw).r;
                        vec4 color = texture(transferTexture, vec2(intensity, 0.5));
                        gl_FragColor = color;
                    }
                `,
                transparent: true
            });

            const plane = new THREE.Mesh(geometry, material);
            this.slicePlanes.push(plane);
            this.scene.add(plane);
            
            return plane;
        }

        updateTransferFunction(tf) {
            this.transferFunction = tf;
            this.createTransferTexture();
            this.volumeMesh.material.uniforms.transferTexture.value = this.transferTexture;
        }
    }

    // Terrain Renderer
    class TerrainRenderer extends Scene3D {
        constructor(container, options = {}) {
            super(container, {
                camera: { position: { x: 0, y: 50, z: 100 } },
                ...options
            });

            this.terrain = null;
            this.water = null;
            this.skybox = null;
        }

        generateTerrain(width, height, segments, options = {}) {
            const geometry = new THREE.PlaneGeometry(width, height, segments, segments);
            
            // Generate heightmap using Perlin-like noise
            const positions = geometry.attributes.position.array;
            const scale = options.scale || 10;
            const octaves = options.octaves || 4;
            const persistence = options.persistence || 0.5;
            const lacunarity = options.lacunarity || 2.0;

            for (let i = 0; i < positions.length; i += 3) {
                const x = positions[i];
                const y = positions[i + 1];
                
                let elevation = 0;
                let amplitude = 1;
                let frequency = 1;
                let maxValue = 0;

                for (let o = 0; o < octaves; o++) {
                    elevation += amplitude * this.noise2D(x * frequency * 0.01, y * frequency * 0.01);
                    maxValue += amplitude;
                    amplitude *= persistence;
                    frequency *= lacunarity;
                }

                positions[i + 2] = (elevation / maxValue) * scale;
            }

            geometry.computeVertexNormals();
            geometry.attributes.position.needsUpdate = true;

            // Terrain material with height-based coloring
            const material = new THREE.ShaderMaterial({
                uniforms: {
                    minHeight: { value: -scale },
                    maxHeight: { value: scale },
                    waterLevel: { value: options.waterLevel || 0 },
                    lightDirection: { value: new THREE.Vector3(1, 1, 1).normalize() }
                },
                vertexShader: `
                    varying vec3 vPosition;
                    varying vec3 vNormal;
                    varying float vElevation;
                    
                    void main() {
                        vPosition = position;
                        vNormal = normalMatrix * normal;
                        vElevation = position.z;
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                `,
                fragmentShader: `
                    uniform float minHeight;
                    uniform float maxHeight;
                    uniform float waterLevel;
                    uniform vec3 lightDirection;
                    
                    varying vec3 vPosition;
                    varying vec3 vNormal;
                    varying float vElevation;
                    
                    void main() {
                        float t = (vElevation - minHeight) / (maxHeight - minHeight);
                        
                        vec3 deepWater = vec3(0.0, 0.1, 0.3);
                        vec3 shallowWater = vec3(0.0, 0.3, 0.5);
                        vec3 sand = vec3(0.76, 0.70, 0.50);
                        vec3 grass = vec3(0.2, 0.5, 0.1);
                        vec3 forest = vec3(0.1, 0.3, 0.1);
                        vec3 rock = vec3(0.4, 0.4, 0.4);
                        vec3 snow = vec3(0.95, 0.95, 0.95);
                        
                        vec3 color;
                        if (t < 0.3) {
                            color = mix(deepWater, shallowWater, t / 0.3);
                        } else if (t < 0.35) {
                            color = mix(shallowWater, sand, (t - 0.3) / 0.05);
                        } else if (t < 0.5) {
                            color = mix(sand, grass, (t - 0.35) / 0.15);
                        } else if (t < 0.7) {
                            color = mix(grass, forest, (t - 0.5) / 0.2);
                        } else if (t < 0.85) {
                            color = mix(forest, rock, (t - 0.7) / 0.15);
                        } else {
                            color = mix(rock, snow, (t - 0.85) / 0.15);
                        }
                        
                        // Lighting
                        float diffuse = max(dot(vNormal, lightDirection), 0.0);
                        float ambient = 0.3;
                        color *= (ambient + diffuse * 0.7);
                        
                        gl_FragColor = vec4(color, 1.0);
                    }
                `
            });

            this.terrain = new THREE.Mesh(geometry, material);
            this.terrain.rotation.x = -Math.PI / 2;
            this.terrain.receiveShadow = true;
            this.scene.add(this.terrain);

            return this.terrain;
        }

        noise2D(x, y) {
            // Simple noise function using sin
            const n = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
            return (n - Math.floor(n)) * 2 - 1;
        }

        addWater(level, size) {
            const geometry = new THREE.PlaneGeometry(size, size, 100, 100);
            
            const material = new THREE.ShaderMaterial({
                uniforms: {
                    time: { value: 0 },
                    waterColor: { value: new THREE.Color(0x0066aa) },
                    foamColor: { value: new THREE.Color(0xffffff) }
                },
                vertexShader: `
                    uniform float time;
                    varying vec2 vUv;
                    varying float vWaveHeight;
                    
                    void main() {
                        vUv = uv;
                        
                        vec3 pos = position;
                        float wave1 = sin(pos.x * 0.5 + time) * 0.3;
                        float wave2 = sin(pos.y * 0.3 + time * 0.7) * 0.2;
                        float wave3 = sin((pos.x + pos.y) * 0.2 + time * 1.3) * 0.15;
                        pos.z = wave1 + wave2 + wave3;
                        vWaveHeight = pos.z;
                        
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
                    }
                `,
                fragmentShader: `
                    uniform vec3 waterColor;
                    uniform vec3 foamColor;
                    varying vec2 vUv;
                    varying float vWaveHeight;
                    
                    void main() {
                        float foam = smoothstep(0.3, 0.5, vWaveHeight);
                        vec3 color = mix(waterColor, foamColor, foam);
                        gl_FragColor = vec4(color, 0.8);
                    }
                `,
                transparent: true
            });

            this.water = new THREE.Mesh(geometry, material);
            this.water.rotation.x = -Math.PI / 2;
            this.water.position.y = level;
            this.scene.add(this.water);

            // Animate water
            this.addAnimation('water', this.water, (obj) => {
                obj.material.uniforms.time.value += 0.016;
            });
        }

        addSkybox() {
            const geometry = new THREE.SphereGeometry(500, 32, 32);
            
            const material = new THREE.ShaderMaterial({
                uniforms: {
                    sunPosition: { value: new THREE.Vector3(100, 100, 100) }
                },
                vertexShader: `
                    varying vec3 vWorldPosition;
                    void main() {
                        vWorldPosition = position;
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                `,
                fragmentShader: `
                    uniform vec3 sunPosition;
                    varying vec3 vWorldPosition;
                    
                    void main() {
                        vec3 direction = normalize(vWorldPosition);
                        vec3 sunDir = normalize(sunPosition);
                        
                        float sunDot = dot(direction, sunDir);
                        float sun = pow(max(sunDot, 0.0), 256.0);
                        
                        // Sky gradient
                        vec3 skyTop = vec3(0.2, 0.4, 0.8);
                        vec3 skyHorizon = vec3(0.7, 0.8, 0.9);
                        float skyGradient = pow(1.0 - abs(direction.y), 3.0);
                        vec3 sky = mix(skyTop, skyHorizon, skyGradient);
                        
                        // Sun
                        vec3 sunColor = vec3(1.0, 0.9, 0.7);
                        sky += sunColor * sun;
                        
                        gl_FragColor = vec4(sky, 1.0);
                    }
                `,
                side: THREE.BackSide
            });

            this.skybox = new THREE.Mesh(geometry, material);
            this.scene.add(this.skybox);
        }
    }

    // CAD Model Viewer
    class CADViewer extends Scene3D {
        constructor(container, options = {}) {
            super(container, {
                camera: { position: { x: 10, y: 10, z: 10 } },
                ...options
            });

            this.models = new Map();
            this.measurements = [];
            this.gridHelper = null;
            this.axisHelper = null;
        }

        setupHelpers() {
            // Grid
            this.gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
            this.scene.add(this.gridHelper);

            // Axes
            this.axisHelper = new THREE.AxesHelper(5);
            this.scene.add(this.axisHelper);
        }

        loadSTEP(file) {
            // STEP file parsing would require external library
            // This is a placeholder for the interface
            console.log('Loading STEP file:', file.name);
        }

        loadSTL(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const geometry = this.parseSTL(e.target.result);
                const material = new THREE.MeshPhongMaterial({
                    color: 0x888888,
                    specular: 0x222222,
                    shininess: 50
                });
                const mesh = new THREE.Mesh(geometry, material);
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                
                this.models.set(file.name, mesh);
                this.scene.add(mesh);
            };
            reader.readAsArrayBuffer(file);
        }

        parseSTL(buffer) {
            const geometry = new THREE.BufferGeometry();
            const data = new DataView(buffer);
            const faces = (buffer.byteLength - 84) / 50;
            const vertices = new Float32Array(faces * 9);
            const normals = new Float32Array(faces * 9);

            let offset = 84;
            for (let i = 0; i < faces; i++) {
                const normal = [
                    data.getFloat32(offset, true),
                    data.getFloat32(offset + 4, true),
                    data.getFloat32(offset + 8, true)
                ];
                offset += 12;

                for (let j = 0; j < 3; j++) {
                    const idx = i * 9 + j * 3;
                    vertices[idx] = data.getFloat32(offset, true);
                    vertices[idx + 1] = data.getFloat32(offset + 4, true);
                    vertices[idx + 2] = data.getFloat32(offset + 8, true);
                    normals[idx] = normal[0];
                    normals[idx + 1] = normal[1];
                    normals[idx + 2] = normal[2];
                    offset += 12;
                }
                offset += 2;
            }

            geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
            return geometry;
        }

        measureDistance(point1, point2) {
            const distance = Math.sqrt(
                Math.pow(point2.x - point1.x, 2) +
                Math.pow(point2.y - point1.y, 2) +
                Math.pow(point2.z - point1.z, 2)
            );

            this.measurements.push({
                type: 'distance',
                start: point1,
                end: point2,
                value: distance
            });

            // Draw measurement line
            const geometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(point1.x, point1.y, point1.z),
                new THREE.Vector3(point2.x, point2.y, point2.z)
            ]);
            const material = new THREE.LineBasicMaterial({ color: 0xffff00 });
            const line = new THREE.Line(geometry, material);
            this.scene.add(line);

            return distance;
        }

        measureAngle(point1, vertex, point2) {
            const v1 = new THREE.Vector3(
                point1.x - vertex.x,
                point1.y - vertex.y,
                point1.z - vertex.z
            ).normalize();
            const v2 = new THREE.Vector3(
                point2.x - vertex.x,
                point2.y - vertex.y,
                point2.z - vertex.z
            ).normalize();

            const angle = Math.acos(v1.dot(v2)) * 180 / Math.PI;

            this.measurements.push({
                type: 'angle',
                point1,
                vertex,
                point2,
                value: angle
            });

            return angle;
        }

        crossSection(plane) {
            const sections = [];
            
            this.models.forEach((model, name) => {
                const geometry = model.geometry;
                const positions = geometry.attributes.position.array;
                const sectionPoints = [];

                for (let i = 0; i < positions.length; i += 9) {
                    const v1 = new THREE.Vector3(positions[i], positions[i + 1], positions[i + 2]);
                    const v2 = new THREE.Vector3(positions[i + 3], positions[i + 4], positions[i + 5]);
                    const v3 = new THREE.Vector3(positions[i + 6], positions[i + 7], positions[i + 8]);

                    const intersections = this.trianglePlaneIntersection(v1, v2, v3, plane);
                    sectionPoints.push(...intersections);
                }

                if (sectionPoints.length > 0) {
                    sections.push({ name, points: sectionPoints });
                }
            });

            return sections;
        }

        trianglePlaneIntersection(v1, v2, v3, plane) {
            const d1 = plane.distanceToPoint(v1);
            const d2 = plane.distanceToPoint(v2);
            const d3 = plane.distanceToPoint(v3);

            const intersections = [];
            const epsilon = 0.0001;

            // Check each edge for intersection
            if (d1 * d2 < 0) {
                const t = d1 / (d1 - d2);
                intersections.push(new THREE.Vector3().lerpVectors(v1, v2, t));
            }
            if (d2 * d3 < 0) {
                const t = d2 / (d2 - d3);
                intersections.push(new THREE.Vector3().lerpVectors(v2, v3, t));
            }
            if (d3 * d1 < 0) {
                const t = d3 / (d3 - d1);
                intersections.push(new THREE.Vector3().lerpVectors(v3, v1, t));
            }

            return intersections;
        }

        exportSTL(name) {
            const model = this.models.get(name);
            if (!model) return null;

            const geometry = model.geometry;
            const positions = geometry.attributes.position.array;
            const normals = geometry.attributes.normal.array;

            let stl = 'solid model\n';
            for (let i = 0; i < positions.length; i += 9) {
                stl += `facet normal ${normals[i]} ${normals[i+1]} ${normals[i+2]}\n`;
                stl += '  outer loop\n';
                stl += `    vertex ${positions[i]} ${positions[i+1]} ${positions[i+2]}\n`;
                stl += `    vertex ${positions[i+3]} ${positions[i+4]} ${positions[i+5]}\n`;
                stl += `    vertex ${positions[i+6]} ${positions[i+7]} ${positions[i+8]}\n`;
                stl += '  endloop\n';
                stl += 'endfacet\n';
            }
            stl += 'endsolid model\n';

            return stl;
        }
    }

    // Animation System
    class AnimationSystem {
        constructor(scene) {
            this.scene = scene;
            this.clips = new Map();
            this.mixer = null;
            this.actions = new Map();
        }

        createClip(name, tracks) {
            const clip = {
                name,
                tracks: [],
                duration: 0
            };

            tracks.forEach(track => {
                clip.tracks.push({
                    target: track.target,
                    property: track.property,
                    times: track.times,
                    values: track.values,
                    interpolation: track.interpolation || 'linear'
                });
                clip.duration = Math.max(clip.duration, ...track.times);
            });

            this.clips.set(name, clip);
            return clip;
        }

        play(clipName, options = {}) {
            const clip = this.clips.get(clipName);
            if (!clip) return;

            const action = {
                clip: clip,
                time: 0,
                speed: options.speed || 1,
                loop: options.loop || false,
                playing: true,
                weight: 1,
                startTime: Date.now()
            };

            this.actions.set(clipName, action);
            this.animate();
        }

        stop(clipName) {
            const action = this.actions.get(clipName);
            if (action) {
                action.playing = false;
                this.actions.delete(clipName);
            }
        }

        pause(clipName) {
            const action = this.actions.get(clipName);
            if (action) {
                action.playing = false;
            }
        }

        resume(clipName) {
            const action = this.actions.get(clipName);
            if (action) {
                action.playing = true;
                action.startTime = Date.now() - action.time * 1000 / action.speed;
            }
        }

        animate() {
            const now = Date.now();

            this.actions.forEach((action, name) => {
                if (!action.playing) return;

                const elapsed = (now - action.startTime) / 1000 * action.speed;
                action.time = elapsed % action.clip.duration;

                if (!action.loop && elapsed >= action.clip.duration) {
                    action.time = action.clip.duration;
                    action.playing = false;
                }

                // Apply tracks
                action.clip.tracks.forEach(track => {
                    this.applyTrack(track, action.time);
                });
            });
        }

        applyTrack(track, time) {
            // Find keyframes
            let i = 0;
            while (i < track.times.length - 1 && track.times[i + 1] < time) {
                i++;
            }

            const t1 = track.times[i];
            const t2 = track.times[i + 1] || t1;
            const alpha = (time - t1) / (t2 - t1 || 1);

            // Interpolate values
            const v1 = track.values[i];
            const v2 = track.values[i + 1] || v1;

            let value;
            if (track.interpolation === 'step') {
                value = v1;
            } else if (track.interpolation === 'linear') {
                value = this.lerp(v1, v2, alpha);
            } else if (track.interpolation === 'spline') {
                value = this.splineInterpolate(track.values, track.times, time, i);
            }

            // Apply to target
            this.applyValue(track.target, track.property, value);
        }

        lerp(v1, v2, alpha) {
            if (typeof v1 === 'number') {
                return v1 + (v2 - v1) * alpha;
            } else if (v1 instanceof THREE.Vector3) {
                return new THREE.Vector3().lerpVectors(v1, v2, alpha);
            } else if (v1 instanceof THREE.Quaternion) {
                return new THREE.Quaternion().slerpQuaternions(v1, v2, alpha);
            } else if (Array.isArray(v1)) {
                return v1.map((v, i) => v + (v2[i] - v) * alpha);
            }
            return v1;
        }

        splineInterpolate(values, times, time, index) {
            // Catmull-Rom spline interpolation
            const p0 = values[Math.max(0, index - 1)];
            const p1 = values[index];
            const p2 = values[Math.min(values.length - 1, index + 1)];
            const p3 = values[Math.min(values.length - 1, index + 2)];

            const t = (time - times[index]) / (times[index + 1] - times[index] || 1);
            const t2 = t * t;
            const t3 = t2 * t;

            // Catmull-Rom coefficients
            const c0 = -0.5 * t3 + t2 - 0.5 * t;
            const c1 = 1.5 * t3 - 2.5 * t2 + 1;
            const c2 = -1.5 * t3 + 2 * t2 + 0.5 * t;
            const c3 = 0.5 * t3 - 0.5 * t2;

            return this.combine(p0, p1, p2, p3, c0, c1, c2, c3);
        }

        combine(p0, p1, p2, p3, c0, c1, c2, c3) {
            if (typeof p0 === 'number') {
                return p0 * c0 + p1 * c1 + p2 * c2 + p3 * c3;
            }
            // Handle other types similarly
            return p1;
        }

        applyValue(target, property, value) {
            const props = property.split('.');
            let obj = target;

            for (let i = 0; i < props.length - 1; i++) {
                obj = obj[props[i]];
            }

            const finalProp = props[props.length - 1];
            if (typeof value === 'object' && value !== null) {
                Object.assign(obj[finalProp], value);
            } else {
                obj[finalProp] = value;
            }
        }

        blendAnimations(clipNames, weights) {
            // Blend multiple animations together
            clipNames.forEach((name, i) => {
                const action = this.actions.get(name);
                if (action) {
                    action.weight = weights[i];
                }
            });
        }
    }

    // Physics Simulation
    class PhysicsSimulation {
        constructor() {
            this.bodies = [];
            this.constraints = [];
            this.gravity = new THREE.Vector3(0, -9.81, 0);
            this.timeStep = 1 / 60;
        }

        addBody(body) {
            body.velocity = body.velocity || new THREE.Vector3();
            body.angularVelocity = body.angularVelocity || new THREE.Vector3();
            body.force = new THREE.Vector3();
            body.torque = new THREE.Vector3();
            body.mass = body.mass || 1;
            body.inertia = body.inertia || 1;
            body.restitution = body.restitution || 0.5;
            body.friction = body.friction || 0.5;
            
            this.bodies.push(body);
            return body;
        }

        addConstraint(constraint) {
            this.constraints.push(constraint);
            return constraint;
        }

        step() {
            // Apply gravity
            this.bodies.forEach(body => {
                if (!body.static) {
                    body.force.addScaledVector(this.gravity, body.mass);
                }
            });

            // Integrate velocities
            this.bodies.forEach(body => {
                if (!body.static) {
                    body.velocity.addScaledVector(body.force, this.timeStep / body.mass);
                    body.angularVelocity.addScaledVector(body.torque, this.timeStep / body.inertia);
                }
            });

            // Solve constraints
            this.constraints.forEach(constraint => {
                this.solveConstraint(constraint);
            });

            // Integrate positions
            this.bodies.forEach(body => {
                if (!body.static) {
                    body.position.addScaledVector(body.velocity, this.timeStep);
                    
                    // Update rotation
                    const angle = body.angularVelocity.length() * this.timeStep;
                    if (angle > 0) {
                        const axis = body.angularVelocity.clone().normalize();
                        const q = new THREE.Quaternion().setFromAxisAngle(axis, angle);
                        body.quaternion.multiply(q);
                    }
                }

                // Clear forces
                body.force.set(0, 0, 0);
                body.torque.set(0, 0, 0);
            });

            // Detect and resolve collisions
            this.detectCollisions();
        }

        solveConstraint(constraint) {
            const body1 = constraint.body1;
            const body2 = constraint.body2;

            switch (constraint.type) {
                case 'distance':
                    this.solveDistanceConstraint(body1, body2, constraint.distance);
                    break;
                case 'hinge':
                    this.solveHingeConstraint(body1, body2, constraint.axis);
                    break;
                case 'ball':
                    this.solveBallConstraint(body1, body2, constraint.anchor);
                    break;
            }
        }

        solveDistanceConstraint(body1, body2, targetDistance) {
            const delta = new THREE.Vector3().subVectors(body2.position, body1.position);
            const currentDistance = delta.length();
            const error = currentDistance - targetDistance;

            if (Math.abs(error) < 0.0001) return;

            const direction = delta.normalize();
            const correction = direction.multiplyScalar(error * 0.5);

            if (!body1.static) {
                body1.position.add(correction);
            }
            if (!body2.static) {
                body2.position.sub(correction);
            }
        }

        solveHingeConstraint(body1, body2, axis) {
            // Align rotation axes
            const axis1 = new THREE.Vector3(1, 0, 0).applyQuaternion(body1.quaternion);
            const axis2 = new THREE.Vector3(1, 0, 0).applyQuaternion(body2.quaternion);
            
            const correction = new THREE.Vector3().crossVectors(axis1, axis2);
            
            if (!body1.static) {
                body1.angularVelocity.addScaledVector(correction, 0.5);
            }
            if (!body2.static) {
                body2.angularVelocity.subScaledVector(correction, 0.5);
            }
        }

        solveBallConstraint(body1, body2, anchor) {
            // Keep anchor point together
            const anchor1 = anchor.clone().applyQuaternion(body1.quaternion).add(body1.position);
            const anchor2 = anchor.clone().applyQuaternion(body2.quaternion).add(body2.position);
            
            const correction = new THREE.Vector3().subVectors(anchor2, anchor1).multiplyScalar(0.5);
            
            if (!body1.static) {
                body1.position.add(correction);
            }
            if (!body2.static) {
                body2.position.sub(correction);
            }
        }

        detectCollisions() {
            for (let i = 0; i < this.bodies.length; i++) {
                for (let j = i + 1; j < this.bodies.length; j++) {
                    const body1 = this.bodies[i];
                    const body2 = this.bodies[j];

                    if (body1.static && body2.static) continue;

                    const collision = this.checkCollision(body1, body2);
                    if (collision) {
                        this.resolveCollision(body1, body2, collision);
                    }
                }
            }
        }

        checkCollision(body1, body2) {
            // Simple sphere collision detection
            if (body1.shape === 'sphere' && body2.shape === 'sphere') {
                const delta = new THREE.Vector3().subVectors(body2.position, body1.position);
                const distance = delta.length();
                const minDist = body1.radius + body2.radius;

                if (distance < minDist) {
                    return {
                        normal: delta.normalize(),
                        penetration: minDist - distance,
                        point: new THREE.Vector3().addVectors(body1.position, body2.position).multiplyScalar(0.5)
                    };
                }
            }
            return null;
        }

        resolveCollision(body1, body2, collision) {
            const { normal, penetration } = collision;

            // Separate bodies
            const separation = normal.clone().multiplyScalar(penetration * 0.5);
            if (!body1.static) body1.position.sub(separation);
            if (!body2.static) body2.position.add(separation);

            // Calculate relative velocity
            const relativeVelocity = new THREE.Vector3().subVectors(body2.velocity, body1.velocity);
            const velocityAlongNormal = relativeVelocity.dot(normal);

            if (velocityAlongNormal > 0) return; // Moving apart

            // Calculate impulse
            const restitution = Math.min(body1.restitution, body2.restitution);
            let j = -(1 + restitution) * velocityAlongNormal;
            j /= (body1.static ? 0 : 1 / body1.mass) + (body2.static ? 0 : 1 / body2.mass);

            const impulse = normal.clone().multiplyScalar(j);

            if (!body1.static) {
                body1.velocity.subScaledVector(impulse, 1 / body1.mass);
            }
            if (!body2.static) {
                body2.velocity.addScaledVector(impulse, 1 / body2.mass);
            }

            // Apply friction
            const tangent = relativeVelocity.clone().subScaledVector(normal, velocityAlongNormal).normalize();
            const frictionImpulse = tangent.multiplyScalar(-j * Math.sqrt(body1.friction * body2.friction));

            if (!body1.static) {
                body1.velocity.sub(frictionImpulse.clone().multiplyScalar(1 / body1.mass));
            }
            if (!body2.static) {
                body2.velocity.add(frictionImpulse.clone().multiplyScalar(1 / body2.mass));
            }
        }
    }

    // Post Processing Effects
    class PostProcessing {
        constructor(renderer, scene, camera) {
            this.renderer = renderer;
            this.scene = scene;
            this.camera = camera;
            this.enabled = true;
            this.passes = [];
            
            this.init();
        }
        
        init() {
            // Create render targets
            const size = this.renderer.getSize(new THREE.Vector2());
            
            this.readTarget = new THREE.WebGLRenderTarget(size.x, size.y, {
                minFilter: THREE.LinearFilter,
                magFilter: THREE.LinearFilter,
                format: THREE.RGBAFormat,
                type: THREE.HalfFloatType
            });
            
            this.writeTarget = this.readTarget.clone();
        }
        
        addPass(pass) {
            this.passes.push(pass);
            pass.setSize(this.readTarget.width, this.readTarget.height);
        }
        
        removePass(pass) {
            const index = this.passes.indexOf(pass);
            if (index > -1) {
                this.passes.splice(index, 1);
            }
        }
        
        render() {
            if (!this.enabled || this.passes.length === 0) {
                this.renderer.render(this.scene, this.camera);
                return;
            }
            
            // Render scene to target
            this.renderer.setRenderTarget(this.readTarget);
            this.renderer.render(this.scene, this.camera);
            
            // Apply passes
            for (let i = 0; i < this.passes.length; i++) {
                const pass = this.passes[i];
                
                if (pass.enabled) {
                    pass.render(
                        this.renderer,
                        this.readTarget,
                        this.writeTarget,
                        i === this.passes.length - 1
                    );
                    
                    // Swap buffers
                    [this.readTarget, this.writeTarget] = [this.writeTarget, this.readTarget];
                }
            }
            
            // Final render to screen
            this.renderer.setRenderTarget(null);
        }
        
        setSize(width, height) {
            this.readTarget.setSize(width, height);
            this.writeTarget.setSize(width, height);
            this.passes.forEach(pass => pass.setSize(width, height));
        }
    }

    // Bloom Effect Pass
    class BloomPass {
        constructor(options = {}) {
            this.enabled = true;
            this.strength = options.strength || 1.5;
            this.radius = options.radius || 0.4;
            this.threshold = options.threshold || 0.8;
            
            this.createMaterials();
        }
        
        createMaterials() {
            // Luminosity high pass material
            this.luminosityMaterial = new THREE.ShaderMaterial({
                uniforms: {
                    tDiffuse: { value: null },
                    luminosityThreshold: { value: this.threshold },
                    smoothWidth: { value: 0.01 }
                },
                vertexShader: `
                    varying vec2 vUv;
                    void main() {
                        vUv = uv;
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                `,
                fragmentShader: `
                    uniform sampler2D tDiffuse;
                    uniform float luminosityThreshold;
                    uniform float smoothWidth;
                    varying vec2 vUv;
                    
                    void main() {
                        vec4 texel = texture2D(tDiffuse, vUv);
                        float v = luminance(texel.rgb);
                        float alpha = smoothstep(luminosityThreshold, luminosityThreshold + smoothWidth, v);
                        gl_FragColor = vec4(texel.rgb * alpha, texel.a * alpha);
                    }
                `
            });
            
            // Gaussian blur material
            this.blurMaterial = new THREE.ShaderMaterial({
                uniforms: {
                    tDiffuse: { value: null },
                    direction: { value: new THREE.Vector2(1, 0) }
                },
                vertexShader: `
                    varying vec2 vUv;
                    void main() {
                        vUv = uv;
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                `,
                fragmentShader: `
                    uniform sampler2D tDiffuse;
                    uniform vec2 direction;
                    varying vec2 vUv;
                    
                    void main() {
                        vec4 sum = vec4(0.0);
                        vec2 texelSize = vec2(1.0 / 512.0);
                        
                        sum += texture2D(tDiffuse, vUv - 4.0 * direction * texelSize) * 0.051;
                        sum += texture2D(tDiffuse, vUv - 3.0 * direction * texelSize) * 0.0918;
                        sum += texture2D(tDiffuse, vUv - 2.0 * direction * texelSize) * 0.12245;
                        sum += texture2D(tDiffuse, vUv - 1.0 * direction * texelSize) * 0.1531;
                        sum += texture2D(tDiffuse, vUv) * 0.1633;
                        sum += texture2D(tDiffuse, vUv + 1.0 * direction * texelSize) * 0.1531;
                        sum += texture2D(tDiffuse, vUv + 2.0 * direction * texelSize) * 0.12245;
                        sum += texture2D(tDiffuse, vUv + 3.0 * direction * texelSize) * 0.0918;
                        sum += texture2D(tDiffuse, vUv + 4.0 * direction * texelSize) * 0.051;
                        
                        gl_FragColor = sum;
                    }
                `
            });
            
            // Composite material
            this.compositeMaterial = new THREE.ShaderMaterial({
                uniforms: {
                    tDiffuse: { value: null },
                    tBloom: { value: null },
                    bloomStrength: { value: this.strength }
                },
                vertexShader: `
                    varying vec2 vUv;
                    void main() {
                        vUv = uv;
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                `,
                fragmentShader: `
                    uniform sampler2D tDiffuse;
                    uniform sampler2D tBloom;
                    uniform float bloomStrength;
                    varying vec2 vUv;
                    
                    void main() {
                        vec4 color = texture2D(tDiffuse, vUv);
                        vec4 bloom = texture2D(tBloom, vUv);
                        gl_FragColor = color + bloom * bloomStrength;
                    }
                `
            });
        }
        
        setSize(width, height) {
            this.width = width;
            this.height = height;
        }
        
        render(renderer, readTarget, writeTarget, isLast) {
            // Implementation would go here
        }
    }

    // SSAO (Screen Space Ambient Occlusion) Pass
    class SSAOPass {
        constructor(options = {}) {
            this.enabled = true;
            this.radius = options.radius || 16;
            this.aoClamp = options.aoClamp || 0.25;
            this.lumInfluence = options.lumInfluence || 0.7;
            
            this.createMaterials();
        }
        
        createMaterials() {
            this.ssaoMaterial = new THREE.ShaderMaterial({
                uniforms: {
                    tDiffuse: { value: null },
                    tDepth: { value: null },
                    resolution: { value: new THREE.Vector2() },
                    radius: { value: this.radius },
                    aoClamp: { value: this.aoClamp },
                    lumInfluence: { value: this.lumInfluence },
                    samples: { value: 32 },
                    rings: { value: 4 },
                    random: { value: null }
                },
                vertexShader: `
                    varying vec2 vUv;
                    void main() {
                        vUv = uv;
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                `,
                fragmentShader: `
                    uniform sampler2D tDiffuse;
                    uniform sampler2D tDepth;
                    uniform vec2 resolution;
                    uniform float radius;
                    uniform float aoClamp;
                    uniform float lumInfluence;
                    uniform int samples;
                    uniform int rings;
                    uniform sampler2D random;
                    
                    varying vec2 vUv;
                    
                    float compareDepths(in float depth1, in float depth2) {
                        float ao = 0.0;
                        float diff = (depth1 - depth2) * 100.0;
                        if (diff < 0.0) {
                            ao = 1.0 - clamp(-diff / aoClamp, 0.0, 1.0);
                        } else {
                            ao = clamp(diff / aoClamp, 0.0, 1.0);
                        }
                        return ao;
                    }
                    
                    void main() {
                        float depth = texture2D(tDepth, vUv).r;
                        float ao = 0.0;
                        
                        vec2 sampleDirection = vec2(0.0);
                        float randomAngle = texture2D(random, vUv * resolution / 64.0).r * 6.28318530718;
                        
                        float aspect = resolution.x / resolution.y;
                        vec2 aspectCorrect = vec2(1.0, aspect);
                        
                        for (int i = 0; i < 4; i++) {
                            for (int j = 0; j < 8; j++) {
                                float angle = float(j) * 0.7853981634 + randomAngle;
                                float r = (float(i) + 0.5) / float(4);
                                
                                sampleDirection = vec2(cos(angle), sin(angle)) * r * radius;
                                sampleDirection = sampleDirection * aspectCorrect;
                                
                                vec2 sampleUv = vUv + sampleDirection / resolution;
                                float sampleDepth = texture2D(tDepth, sampleUv).r;
                                
                                ao += compareDepths(depth, sampleDepth);
                            }
                        }
                        
                        ao /= 32.0;
                        ao = clamp(ao, 0.0, 1.0);
                        
                        vec4 color = texture2D(tDiffuse, vUv);
                        float lum = dot(color.rgb, vec3(0.299, 0.587, 0.114));
                        
                        gl_FragColor = vec4(color.rgb * mix(vec3(1.0), vec3(ao), lumInfluence), color.a);
                    }
                `
            });
        }
        
        setSize(width, height) {
            this.ssaoMaterial.uniforms.resolution.value.set(width, height);
        }
        
        render(renderer, readTarget, writeTarget, isLast) {
            // Implementation would go here
        }
    }

    // Instanced Mesh Manager
    class InstancedMeshManager {
        constructor(scene, geometry, material, maxInstances = 10000) {
            this.scene = scene;
            this.geometry = geometry;
            this.material = material;
            this.maxInstances = maxInstances;
            
            this.instances = [];
            this.instanceMap = new Map();
            this.nextId = 0;
            
            this.createInstancedMesh();
        }
        
        createInstancedMesh() {
            this.instancedMesh = new THREE.InstancedMesh(
                this.geometry,
                this.material,
                this.maxInstances
            );
            
            this.instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
            this.scene.add(this.instancedMesh);
            
            this.dummy = new THREE.Object3D();
        }
        
        addInstance(position, rotation, scale, data = {}) {
            if (this.instances.length >= this.maxInstances) {
                console.warn('Maximum instances reached');
                return null;
            }
            
            const id = this.nextId++;
            const instance = {
                id,
                position: position.clone ? position.clone() : new THREE.Vector3(position.x, position.y, position.z),
                rotation: rotation.clone ? rotation.clone() : new THREE.Euler(rotation.x, rotation.y, rotation.z),
                scale: scale.clone ? scale.clone() : new THREE.Vector3(scale.x || scale, scale.y || scale, scale.z || scale),
                data,
                visible: true
            };
            
            this.instances.push(instance);
            this.instanceMap.set(id, instance);
            
            this.updateInstance(instance);
            
            return id;
        }
        
        removeInstance(id) {
            const instance = this.instanceMap.get(id);
            if (!instance) return false;
            
            const index = this.instances.indexOf(instance);
            if (index > -1) {
                this.instances.splice(index, 1);
                this.instanceMap.delete(id);
                
                // Shift all instances after this one
                for (let i = index; i < this.instances.length; i++) {
                    this.updateInstanceAtIndex(this.instances[i], i);
                }
                
                this.instancedMesh.count = this.instances.length;
                this.instancedMesh.instanceMatrix.needsUpdate = true;
            }
            
            return true;
        }
        
        updateInstance(instance) {
            const index = this.instances.indexOf(instance);
            if (index === -1) return;
            
            this.updateInstanceAtIndex(instance, index);
        }
        
        updateInstanceAtIndex(instance, index) {
            this.dummy.position.copy(instance.position);
            this.dummy.rotation.copy(instance.rotation);
            this.dummy.scale.copy(instance.scale);
            this.dummy.updateMatrix();
            
            this.instancedMesh.setMatrixAt(index, this.dummy.matrix);
            this.instancedMesh.instanceMatrix.needsUpdate = true;
        }
        
        updateInstancePosition(id, position) {
            const instance = this.instanceMap.get(id);
            if (!instance) return;
            
            instance.position.copy(position);
            this.updateInstance(instance);
        }
        
        updateInstanceRotation(id, rotation) {
            const instance = this.instanceMap.get(id);
            if (!instance) return;
            
            instance.rotation.copy(rotation);
            this.updateInstance(instance);
        }
        
        updateInstanceScale(id, scale) {
            const instance = this.instanceMap.get(id);
            if (!instance) return;
            
            instance.scale.copy(scale);
            this.updateInstance(instance);
        }
        
        setInstanceVisibility(id, visible) {
            const instance = this.instanceMap.get(id);
            if (!instance) return;
            
            instance.visible = visible;
            this.updateInstance(instance);
        }
        
        getInstance(id) {
            return this.instanceMap.get(id);
        }
        
        getAllInstances() {
            return [...this.instances];
        }
        
        raycast(raycaster) {
            const intersects = [];
            const matrix = new THREE.Matrix4();
            const inverseMatrix = new THREE.Matrix4();
            
            for (let i = 0; i < this.instances.length; i++) {
                const instance = this.instances[i];
                if (!instance.visible) continue;
                
                this.instancedMesh.getMatrixAt(i, matrix);
                inverseMatrix.copy(matrix).invert();
                
                const ray = raycaster.ray.clone();
                ray.applyMatrix4(inverseMatrix);
                
                const intersection = ray.intersectBox(this.geometry.boundingBox, new THREE.Vector3());
                if (intersection) {
                    intersects.push({
                        instance,
                        instanceId: instance.id,
                        point: intersection.applyMatrix4(matrix),
                        distance: raycaster.ray.origin.distanceTo(intersection.applyMatrix4(matrix))
                    });
                }
            }
            
            intersects.sort((a, b) => a.distance - b.distance);
            return intersects;
        }
        
        dispose() {
            this.scene.remove(this.instancedMesh);
            this.instancedMesh.dispose();
            this.instances = [];
            this.instanceMap.clear();
        }
    }

    // LOD (Level of Detail) Manager
    class LODManager {
        constructor(camera) {
            this.camera = camera;
            this.lodGroups = new Map();
            this.distanceThresholds = [10, 50, 100, 200];
        }
        
        createLODGroup(id, levels) {
            // levels = [{ distance: 10, mesh: mesh1 }, { distance: 50, mesh: mesh2 }, ...]
            const lodGroup = {
                id,
                levels: levels.sort((a, b) => a.distance - b.distance),
                currentLevel: 0,
                position: new THREE.Vector3()
            };
            
            this.lodGroups.set(id, lodGroup);
            return lodGroup;
        }
        
        updateLODGroup(id, position) {
            const lodGroup = this.lodGroups.get(id);
            if (!lodGroup) return;
            
            lodGroup.position.copy(position);
            
            const distance = this.camera.position.distanceTo(position);
            let newLevel = 0;
            
            for (let i = 0; i < lodGroup.levels.length; i++) {
                if (distance >= lodGroup.levels[i].distance) {
                    newLevel = i;
                }
            }
            
            if (newLevel !== lodGroup.currentLevel) {
                lodGroup.levels[lodGroup.currentLevel].mesh.visible = false;
                lodGroup.levels[newLevel].mesh.visible = true;
                lodGroup.currentLevel = newLevel;
            }
        }
        
        updateAll() {
            this.lodGroups.forEach((lodGroup, id) => {
                this.updateLODGroup(id, lodGroup.position);
            });
        }
        
        setDistanceThresholds(thresholds) {
            this.distanceThresholds = thresholds.sort((a, b) => a - b);
        }
        
        removeLODGroup(id) {
            this.lodGroups.delete(id);
        }
        
        getLODStats() {
            const stats = {
                total: this.lodGroups.size,
                byLevel: {}
            };
            
            this.lodGroups.forEach(lodGroup => {
                const level = lodGroup.currentLevel;
                stats.byLevel[level] = (stats.byLevel[level] || 0) + 1;
            });
            
            return stats;
        }
    }

    // Frustum Culling Helper
    class FrustumCuller {
        constructor(camera) {
            this.camera = camera;
            this.frustum = new THREE.Frustum();
            this.projScreenMatrix = new THREE.Matrix4();
        }
        
        update() {
            this.projScreenMatrix.multiplyMatrices(
                this.camera.projectionMatrix,
                this.camera.matrixWorldInverse
            );
            this.frustum.setFromProjectionMatrix(this.projScreenMatrix);
        }
        
        isVisible(object) {
            if (object.geometry && object.geometry.boundingSphere) {
                const sphere = object.geometry.boundingSphere.clone();
                sphere.applyMatrix4(object.matrixWorld);
                return this.frustum.intersectsSphere(sphere);
            }
            return true;
        }
        
        cullObjects(objects) {
            this.update();
            
            const visible = [];
            const culled = [];
            
            objects.forEach(object => {
                if (this.isVisible(object)) {
                    visible.push(object);
                    object.visible = true;
                } else {
                    culled.push(object);
                    object.visible = false;
                }
            });
            
            return { visible, culled };
        }
        
        getVisibleCount(objects) {
            this.update();
            return objects.filter(obj => this.isVisible(obj)).length;
        }
    }

    // Octree for spatial partitioning
    class Octree {
        constructor(bounds, maxDepth = 8, maxObjects = 8) {
            this.bounds = bounds;
            this.maxDepth = maxDepth;
            this.maxObjects = maxObjects;
            this.objects = [];
            this.children = null;
            this.depth = 0;
        }
        
        insert(object, bounds) {
            if (!this.contains(bounds)) {
                return false;
            }
            
            if (this.children === null && this.objects.length < this.maxObjects) {
                this.objects.push({ object, bounds });
                return true;
            }
            
            if (this.children === null) {
                this.subdivide();
            }
            
            for (let i = 0; i < 8; i++) {
                if (this.children[i].insert(object, bounds)) {
                    return true;
                }
            }
            
            this.objects.push({ object, bounds });
            return true;
        }
        
        subdivide() {
            const min = this.bounds.min;
            const max = this.bounds.max;
            const mid = new THREE.Vector3().addVectors(min, max).multiplyScalar(0.5);
            
            this.children = [];
            
            for (let i = 0; i < 8; i++) {
                const childMin = new THREE.Vector3(
                    (i & 1) ? mid.x : min.x,
                    (i & 2) ? mid.y : min.y,
                    (i & 4) ? mid.z : min.z
                );
                const childMax = new THREE.Vector3(
                    (i & 1) ? max.x : mid.x,
                    (i & 2) ? max.y : mid.y,
                    (i & 4) ? max.z : mid.z
                );
                
                const child = new Octree(
                    { min: childMin, max: childMax },
                    this.maxDepth,
                    this.maxObjects
                );
                child.depth = this.depth + 1;
                this.children.push(child);
            }
            
            // Re-insert objects
            const objects = this.objects;
            this.objects = [];
            
            objects.forEach(({ object, bounds }) => {
                let inserted = false;
                for (let i = 0; i < 8; i++) {
                    if (this.children[i].insert(object, bounds)) {
                        inserted = true;
                        break;
                    }
                }
                if (!inserted) {
                    this.objects.push({ object, bounds });
                }
            });
        }
        
        contains(bounds) {
            return (
                bounds.min.x >= this.bounds.min.x &&
                bounds.min.y >= this.bounds.min.y &&
                bounds.min.z >= this.bounds.min.z &&
                bounds.max.x <= this.bounds.max.x &&
                bounds.max.y <= this.bounds.max.y &&
                bounds.max.z <= this.bounds.max.z
            );
        }
        
        intersects(bounds) {
            return (
                bounds.min.x <= this.bounds.max.x &&
                bounds.max.x >= this.bounds.min.x &&
                bounds.min.y <= this.bounds.max.y &&
                bounds.max.y >= this.bounds.min.y &&
                bounds.min.z <= this.bounds.max.z &&
                bounds.max.z >= this.bounds.min.z
            );
        }
        
        search(bounds) {
            const results = [];
            
            if (!this.intersects(bounds)) {
                return results;
            }
            
            this.objects.forEach(({ object, bounds: objBounds }) => {
                if (this.boundsIntersect(bounds, objBounds)) {
                    results.push(object);
                }
            });
            
            if (this.children !== null) {
                for (let i = 0; i < 8; i++) {
                    results.push(...this.children[i].search(bounds));
                }
            }
            
            return results;
        }
        
        boundsIntersect(a, b) {
            return (
                a.min.x <= b.max.x && a.max.x >= b.min.x &&
                a.min.y <= b.max.y && a.max.y >= b.min.y &&
                a.min.z <= b.max.z && a.max.z >= b.min.z
            );
        }
        
        raycast(ray) {
            const results = [];
            
            if (!this.rayIntersectsBounds(ray)) {
                return results;
            }
            
            this.objects.forEach(({ object, bounds }) => {
                results.push(object);
            });
            
            if (this.children !== null) {
                for (let i = 0; i < 8; i++) {
                    results.push(...this.children[i].raycast(ray));
                }
            }
            
            return results;
        }
        
        rayIntersectsBounds(ray) {
            const min = this.bounds.min;
            const max = this.bounds.max;
            
            let tmin = (min.x - ray.origin.x) / ray.direction.x;
            let tmax = (max.x - ray.origin.x) / ray.direction.x;
            
            if (tmin > tmax) [tmin, tmax] = [tmax, tmin];
            
            let tymin = (min.y - ray.origin.y) / ray.direction.y;
            let tymax = (max.y - ray.origin.y) / ray.direction.y;
            
            if (tymin > tymax) [tymin, tymax] = [tymax, tymin];
            
            if (tmin > tymax || tymin > tmax) return false;
            
            tmin = Math.max(tmin, tymin);
            tmax = Math.min(tmax, tymax);
            
            let tzmin = (min.z - ray.origin.z) / ray.direction.z;
            let tzmax = (max.z - ray.origin.z) / ray.direction.z;
            
            if (tzmin > tzmax) [tzmin, tzmax] = [tzmax, tzmin];
            
            if (tmin > tzmax || tzmin > tmax) return false;
            
            return true;
        }
        
        clear() {
            this.objects = [];
            this.children = null;
        }
        
        getStats() {
            let totalObjects = this.objects.length;
            let totalNodes = 1;
            let maxDepth = this.depth;
            
            if (this.children !== null) {
                for (let i = 0; i < 8; i++) {
                    const childStats = this.children[i].getStats();
                    totalObjects += childStats.totalObjects;
                    totalNodes += childStats.totalNodes;
                    maxDepth = Math.max(maxDepth, childStats.maxDepth);
                }
            }
            
            return { totalObjects, totalNodes, maxDepth };
        }
    }

    // GPU-based Particle Compute System
    class GPUComputeSystem {
        constructor(renderer, options = {}) {
            this.renderer = renderer;
            this.computeShaders = new Map();
            this.buffers = new Map();
            this.options = {
                maxParticles: options.maxParticles || 100000,
                workGroupSize: options.workGroupSize || 256,
                ...options
            };
            
            this.init();
        }
        
        init() {
            // Create compute shader for particle physics
            this.createComputeShader('particlePhysics', `
                struct Particle {
                    position: vec3<f32>,
                    velocity: vec3<f32>,
                    life: f32,
                    size: f32
                }
                
                @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
                @group(0) @binding(1) var<uniform> deltaTime: f32;
                @group(0) @binding(2) var<uniform> gravity: vec3<f32>;
                
                @compute @workgroup_size(${this.options.workGroupSize})
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let index = id.x;
                    var p = particles[index];
                    
                    if (p.life > 0.0) {
                        p.velocity = p.velocity + gravity * deltaTime;
                        p.position = p.position + p.velocity * deltaTime;
                        p.life = p.life - deltaTime;
                        
                        particles[index] = p;
                    }
                }
            `);
            
            // Create compute shader for collision detection
            this.createComputeShader('collision', `
                struct Particle {
                    position: vec3<f32>,
                    velocity: vec3<f32>,
                    life: f32,
                    size: f32
                }
                
                struct CollisionGrid {
                    cells: array<u32>
                }
                
                @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
                @group(0) @binding(1) var<storage, read> grid: CollisionGrid;
                
                @compute @workgroup_size(${this.options.workGroupSize})
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let index = id.x;
                    // Collision resolution logic
                }
            `);
        }
        
        createComputeShader(name, code) {
            this.computeShaders.set(name, {
                code: code,
                pipeline: null,
                bindGroup: null
            });
        }
        
        createBuffer(name, size, usage = 'storage') {
            const buffer = new Float32Array(size);
            this.buffers.set(name, {
                data: buffer,
                usage: usage,
                dirty: true
            });
            return buffer;
        }
        
        updateBuffer(name, data) {
            const buffer = this.buffers.get(name);
            if (buffer) {
                buffer.data.set(data);
                buffer.dirty = true;
            }
        }
        
        dispatch(shaderName, count) {
            const shader = this.computeShaders.get(shaderName);
            if (!shader || !shader.pipeline) return;
            
            const groups = Math.ceil(count / this.options.workGroupSize);
            // Dispatch compute shader
            // Implementation depends on WebGPU API
        }
        
        destroy() {
            this.computeShaders.clear();
            this.buffers.clear();
        }
    }

    // Mesh Simplification System (Quadric Error Metrics)
    class MeshSimplifier {
        constructor() {
            this.threshold = 0.01;
            this.maxIterations = 100;
        }
        
        simplify(geometry, targetRatio = 0.5) {
            const positions = geometry.attributes.position.array;
            const indices = geometry.index ? geometry.index.array : null;
            
            if (!indices) return geometry;
            
            const targetCount = Math.floor(indices.length * targetRatio);
            let currentIndices = new Uint32Array(indices);
            
            // Build vertex adjacency
            const adjacency = this.buildAdjacency(positions, currentIndices);
            
            // Compute quadric errors for each vertex
            const quadrics = this.computeQuadrics(positions, currentIndices);
            
            // Build edge collapse priority queue
            const edges = this.buildEdgeQueue(positions, currentIndices, quadrics);
            
            let iterations = 0;
            while (currentIndices.length > targetCount && iterations < this.maxIterations && edges.length > 0) {
                // Get edge with minimum error
                const edge = edges.shift();
                
                // Check if edge is still valid
                if (!this.isEdgeValid(edge, currentIndices)) continue;
                
                // Collapse edge
                currentIndices = this.collapseEdge(edge, currentIndices, positions);
                
                // Update quadrics
                this.updateQuadrics(edge, quadrics);
                
                // Rebuild affected edges
                this.updateAffectedEdges(edge, edges, adjacency, quadrics);
                
                iterations++;
            }
            
            // Create simplified geometry
            const simplified = new THREE.BufferGeometry();
            simplified.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            simplified.setIndex(new THREE.BufferAttribute(currentIndices, 1));
            simplified.computeVertexNormals();
            
            return simplified;
        }
        
        buildAdjacency(positions, indices) {
            const adjacency = new Map();
            
            for (let i = 0; i < indices.length; i += 3) {
                const a = indices[i], b = indices[i + 1], c = indices[i + 2];
                
                this.addEdge(adjacency, a, b);
                this.addEdge(adjacency, b, c);
                this.addEdge(adjacency, c, a);
            }
            
            return adjacency;
        }
        
        addEdge(adjacency, a, b) {
            if (!adjacency.has(a)) adjacency.set(a, new Set());
            if (!adjacency.has(b)) adjacency.set(b, new Set());
            adjacency.get(a).add(b);
            adjacency.get(b).add(a);
        }
        
        computeQuadrics(positions, indices) {
            const quadrics = new Map();
            
            for (let i = 0; i < positions.length / 3; i++) {
                quadrics.set(i, this.identityQuadric());
            }
            
            for (let i = 0; i < indices.length; i += 3) {
                const a = indices[i], b = indices[i + 1], c = indices[i + 2];
                
                const va = new THREE.Vector3(positions[a * 3], positions[a * 3 + 1], positions[a * 3 + 2]);
                const vb = new THREE.Vector3(positions[b * 3], positions[b * 3 + 1], positions[b * 3 + 2]);
                const vc = new THREE.Vector3(positions[c * 3], positions[c * 3 + 1], positions[c * 3 + 2]);
                
                const plane = this.computePlane(va, vb, vc);
                const Kp = this.planeQuadric(plane);
                
                this.addQuadric(quadrics.get(a), Kp);
                this.addQuadric(quadrics.get(b), Kp);
                this.addQuadric(quadrics.get(c), Kp);
            }
            
            return quadrics;
        }
        
        computePlane(va, vb, vc) {
            const v1 = new THREE.Vector3().subVectors(vb, va);
            const v2 = new THREE.Vector3().subVectors(vc, va);
            const normal = new THREE.Vector3().crossVectors(v1, v2).normalize();
            const d = -normal.dot(va);
            return { normal, d };
        }
        
        planeQuadric(plane) {
            const n = plane.normal;
            const d = plane.d;
            return [
                n.x * n.x, n.x * n.y, n.x * n.z, n.x * d,
                n.y * n.x, n.y * n.y, n.y * n.z, n.y * d,
                n.z * n.x, n.z * n.y, n.z * n.z, n.z * d,
                d * n.x, d * n.y, d * n.z, d * d
            ];
        }
        
        identityQuadric() {
            return new Array(16).fill(0);
        }
        
        addQuadric(target, source) {
            for (let i = 0; i < 16; i++) {
                target[i] += source[i];
            }
        }
        
        buildEdgeQueue(positions, indices, quadrics) {
            const edges = [];
            const seen = new Set();
            
            for (let i = 0; i < indices.length; i += 3) {
                const triangles = [[indices[i], indices[i + 1]], [indices[i + 1], indices[i + 2]], [indices[i + 2], indices[i]]];
                
                for (const [a, b] of triangles) {
                    const key = a < b ? `${a}-${b}` : `${b}-${a}`;
                    if (seen.has(key)) continue;
                    seen.add(key);
                    
                    const error = this.computeEdgeError(a, b, positions, quadrics);
                    edges.push({ a, b, error });
                }
            }
            
            edges.sort((e1, e2) => e1.error - e2.error);
            return edges;
        }
        
        computeEdgeError(a, b, positions, quadrics) {
            const Qa = quadrics.get(a);
            const Qb = quadrics.get(b);
            const Q = this.identityQuadric();
            this.addQuadric(Q, Qa);
            this.addQuadric(Q, Qb);
            
            // Optimal vertex position
            const va = new THREE.Vector3(positions[a * 3], positions[a * 3 + 1], positions[a * 3 + 2]);
            const vb = new THREE.Vector3(positions[b * 3], positions[b * 3 + 1], positions[b * 3 + 2]);
            const v = new THREE.Vector3().addVectors(va, vb).multiplyScalar(0.5);
            
            return this.vertexError(Q, v);
        }
        
        vertexError(Q, v) {
            return Q[0] * v.x * v.x + 2 * Q[1] * v.x * v.y + 2 * Q[2] * v.x * v.z + 2 * Q[3] * v.x +
                   Q[5] * v.y * v.y + 2 * Q[6] * v.y * v.z + 2 * Q[7] * v.y +
                   Q[10] * v.z * v.z + 2 * Q[11] * v.z +
                   Q[15];
        }
        
        isEdgeValid(edge, indices) {
            for (let i = 0; i < indices.length; i++) {
                if (indices[i] === edge.a || indices[i] === edge.b) return true;
            }
            return false;
        }
        
        collapseEdge(edge, indices, positions) {
            const newIndices = [];
            
            for (let i = 0; i < indices.length; i += 3) {
                const a = indices[i], b = indices[i + 1], c = indices[i + 2];
                
                const hasA = a === edge.a || a === edge.b;
                const hasB = b === edge.a || b === edge.b;
                const hasC = c === edge.a || c === edge.b;
                
                const count = (hasA ? 1 : 0) + (hasB ? 1 : 0) + (hasC ? 1 : 0);
                
                if (count < 2) {
                    newIndices.push(
                        a === edge.b ? edge.a : a,
                        b === edge.b ? edge.a : b,
                        c === edge.b ? edge.a : c
                    );
                }
            }
            
            return new Uint32Array(newIndices);
        }
        
        updateQuadrics(edge, quadrics) {
            // Merge quadrics for collapsed vertices
        }
        
        updateAffectedEdges(edge, edges, adjacency, quadrics) {
            // Re-compute errors for affected edges
        }
    }

    // Export
    global.AGI3D = {
        Scene3D,
        MoleculeVisualizer,
        PointCloudVisualizer,
        NeuralNetworkVisualizer,
        ParticleSystemVisualizer,
        VolumeRenderer,
        TerrainRenderer,
        CADViewer,
        AnimationSystem,
        PhysicsSimulation,
        PostProcessing,
        BloomPass,
        SSAOPass,
        InstancedMeshManager,
        LODManager,
        FrustumCuller,
        Octree,
        GPUComputeSystem,
        MeshSimplifier
    };

})(window);
